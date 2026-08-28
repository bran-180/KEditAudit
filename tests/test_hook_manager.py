from collections.abc import Callable
from dataclasses import dataclass

import pytest

from kedit_audit.causal import (
    HookCleanupError,
    HookManager,
    HookRegistrationError,
)

ForwardHook = Callable[[object, tuple[object, ...], object], object | None]


@dataclass(frozen=True)
class ModelOutputLike:
    logits: tuple[float, ...]
    hidden_states: tuple[object, ...]


class FakeHandle:
    def __init__(
        self,
        *,
        hook_id: int,
        remove_callback: Callable[[int], None],
        removal_log: list[int],
        fail: bool = False,
    ) -> None:
        self.hook_id = hook_id
        self._remove_callback = remove_callback
        self._removal_log = removal_log
        self._fail = fail
        self.removed = False

    def remove(self) -> None:
        self._removal_log.append(self.hook_id)
        if self._fail:
            raise RuntimeError(f"remove failed for hook {self.hook_id}")
        if not self.removed:
            self._remove_callback(self.hook_id)
            self.removed = True


class FakeModule:
    def __init__(self, *, failing_handle_ids: set[int] | None = None) -> None:
        self.hooks: dict[int, ForwardHook] = {}
        self.removal_log: list[int] = []
        self._next_id = 0
        self._failing_handle_ids = failing_handle_ids or set()

    def register_forward_hook(self, hook: ForwardHook) -> FakeHandle:
        hook_id = self._next_id
        self._next_id += 1
        self.hooks[hook_id] = hook
        return FakeHandle(
            hook_id=hook_id,
            remove_callback=self.hooks.pop,
            removal_log=self.removal_log,
            fail=hook_id in self._failing_handle_ids,
        )

    def forward(self, output: object) -> object:
        current = output
        for hook in tuple(self.hooks.values()):
            replacement = hook(self, (), current)
            if replacement is not None:
                current = replacement
        return current


def _observer(_module: object, _inputs: tuple[object, ...], _output: object) -> None:
    return None


def test_context_removes_hooks_in_reverse_registration_order() -> None:
    module = FakeModule()

    with HookManager() as manager:
        manager.register_forward_hook(module, _observer)
        manager.register_forward_hook(module, _observer)
        manager.register_forward_hook(module, _observer)
        assert manager.active_hook_count == 3
        assert len(module.hooks) == 3

    assert manager.closed
    assert manager.active_hook_count == 0
    assert module.hooks == {}
    assert module.removal_log == [2, 1, 0]


def test_observer_hook_preserves_tuple_and_model_output_types() -> None:
    module = FakeModule()
    tuple_output = (object(), {"cache": object()})
    model_output = ModelOutputLike((1.0, 2.0), (object(),))

    with HookManager() as manager:
        manager.register_forward_hook(module, _observer)
        assert module.forward(tuple_output) is tuple_output
        assert module.forward(model_output) is model_output


def test_replacement_hook_returns_exact_replacement_type() -> None:
    module = FakeModule()
    replacement = ModelOutputLike((3.0,), ())

    def replace(
        _module: object,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> ModelOutputLike:
        return replacement

    with HookManager() as manager:
        manager.register_forward_hook(module, replace)
        assert module.forward(("original",)) is replacement


def test_exception_path_removes_every_hook_and_preserves_original_error() -> None:
    module = FakeModule()

    with pytest.raises(RuntimeError, match="model forward failed"), HookManager() as manager:
        manager.register_forward_hook(module, _observer)
        manager.register_forward_hook(module, _observer)
        raise RuntimeError("model forward failed")

    assert manager.closed
    assert module.hooks == {}
    assert module.removal_log == [1, 0]


def test_cleanup_failure_attempts_all_handles() -> None:
    module = FakeModule(failing_handle_ids={1})
    manager = HookManager()
    manager.register_forward_hook(module, _observer)
    manager.register_forward_hook(module, _observer)
    manager.register_forward_hook(module, _observer)

    with pytest.raises(HookCleanupError, match="hook 1") as error:
        manager.close()

    assert len(error.value.errors) == 1
    assert module.removal_log == [2, 1, 0]
    assert manager.closed
    assert manager.active_hook_count == 0


def test_cleanup_failure_does_not_mask_model_error() -> None:
    module = FakeModule(failing_handle_ids={0})

    with (
        pytest.raises(RuntimeError, match="primary failure") as error,
        HookManager() as manager,
    ):
        manager.register_forward_hook(module, _observer)
        raise RuntimeError("primary failure")

    assert any("Hook cleanup failed" in note for note in error.value.__notes__)
    assert module.removal_log == [0]


def test_close_is_idempotent_and_closed_manager_rejects_registration() -> None:
    module = FakeModule()
    manager = HookManager()
    manager.register_forward_hook(module, _observer)

    manager.close()
    manager.close()

    with pytest.raises(HookRegistrationError, match="closed"):
        manager.register_forward_hook(module, _observer)
