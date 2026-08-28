"""Forward-hook registration with deterministic, exception-safe cleanup."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, Self

ForwardHook = Callable[[object, tuple[object, ...], object], object | None]


class RemovableHookHandle(Protocol):
    """Minimal handle returned by a forward-hook registrar."""

    def remove(self) -> None:
        """Remove the registered hook."""


class ForwardHookModule(Protocol):
    """Minimal module surface required by HookManager."""

    def register_forward_hook(self, hook: ForwardHook) -> RemovableHookHandle:
        """Register a forward hook and return its removable handle."""


class HookRegistrationError(RuntimeError):
    """Raised when a manager cannot safely register another hook."""


class HookCleanupError(RuntimeError):
    """Raised after cleanup attempted every handle but one or more removals failed."""

    def __init__(self, failures: Sequence[tuple[int, Exception]]) -> None:
        if not failures:
            raise ValueError("HookCleanupError requires at least one failure")
        self.failures = tuple(failures)
        self.errors = tuple(error for _, error in failures)
        details = "; ".join(
            f"hook {index}: {type(error).__name__}: {error}" for index, error in failures
        )
        super().__init__(f"Hook cleanup failed for {details}")


class HookManager:
    """Own forward hooks and remove them in reverse registration order."""

    def __init__(self) -> None:
        self._handles: list[tuple[int, RemovableHookHandle]] = []
        self._next_registration_index = 0
        self._closed = False
        self._entered = False

    @property
    def active_hook_count(self) -> int:
        """Return how many handles remain owned by this manager."""

        return len(self._handles)

    @property
    def closed(self) -> bool:
        """Return whether cleanup has already been attempted."""

        return self._closed

    def register_forward_hook(
        self,
        module: ForwardHookModule,
        hook: ForwardHook,
    ) -> RemovableHookHandle:
        """Register and own one hook until context exit or explicit close."""

        if self._closed:
            raise HookRegistrationError("cannot register a hook on a closed HookManager")
        if not callable(hook):
            raise TypeError("hook must be callable")
        try:
            handle = module.register_forward_hook(hook)
        except Exception as error:
            raise HookRegistrationError("module rejected forward-hook registration") from error
        if not callable(getattr(handle, "remove", None)):
            raise HookRegistrationError("forward-hook registrar returned a non-removable handle")

        registration_index = self._next_registration_index
        self._next_registration_index += 1
        self._handles.append((registration_index, handle))
        return handle

    def close(self) -> None:
        """Attempt every removal once, in reverse registration order."""

        if self._closed:
            return
        self._closed = True
        handles = tuple(reversed(self._handles))
        self._handles.clear()

        failures: list[tuple[int, Exception]] = []
        for registration_index, handle in handles:
            try:
                handle.remove()
            except Exception as error:  # noqa: BLE001 - every remaining handle must be tried.
                failures.append((registration_index, error))
        if failures:
            raise HookCleanupError(failures)

    def __enter__(self) -> Self:
        if self._closed:
            raise HookRegistrationError("cannot enter a closed HookManager")
        if self._entered:
            raise HookRegistrationError("cannot enter the same HookManager more than once")
        self._entered = True
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        exception: BaseException | None,
        _traceback: object,
    ) -> Literal[False]:
        try:
            self.close()
        except HookCleanupError as cleanup_error:
            if exception is None:
                raise
            exception.add_note(str(cleanup_error))
        return False
