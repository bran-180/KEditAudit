from collections import OrderedDict

import pytest

from kedit_audit.adapters import ModulePathError, resolve_module_path


class ModuleNode:
    def __init__(self, **children: object) -> None:
        self._modules = OrderedDict(children)


def _module_tree() -> tuple[ModuleNode, object, object]:
    mlp_zero = object()
    mlp_one = object()
    layers = ModuleNode(
        **{
            "0": ModuleNode(mlp=mlp_zero),
            "1": ModuleNode(mlp=mlp_one),
        }
    )
    root = ModuleNode(transformer=ModuleNode(h=layers))
    return root, mlp_zero, mlp_one


def test_resolves_registered_modules_with_numeric_modulelist_segments() -> None:
    root, mlp_zero, mlp_one = _module_tree()

    assert resolve_module_path(root, "transformer.h.0.mlp") is mlp_zero
    assert resolve_module_path(root, "transformer.h.1.mlp") is mlp_one


def test_resolves_mapping_and_sequence_nodes_without_torch() -> None:
    first = object()
    second = object()
    root = {"layers": [first, second]}

    assert resolve_module_path(root, "layers.0") is first
    assert resolve_module_path(root, "layers.1") is second


def test_resolves_plain_instance_fields_without_invoking_descriptors() -> None:
    leaf = object()

    class PlainNode:
        def __init__(self) -> None:
            self.child = leaf

        @property
        def dangerous(self) -> object:
            raise AssertionError("resolver must not execute descriptors")

    root = PlainNode()

    assert resolve_module_path(root, "child") is leaf
    with pytest.raises(ModulePathError, match="segment 'dangerous' was not found"):
        resolve_module_path(root, "dangerous")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "must not be empty"),
        ("transformer..h", "empty segment"),
        ("transformer._modules", "private segment"),
        ("transformer.h.01", "leading zero"),
        ("transformer.h.-1", "invalid segment"),
    ],
)
def test_rejects_ambiguous_or_private_paths(path: str, message: str) -> None:
    root, _, _ = _module_tree()

    with pytest.raises(ModulePathError, match=message):
        resolve_module_path(root, path)


def test_missing_segment_reports_resolved_prefix() -> None:
    root, _, _ = _module_tree()

    with pytest.raises(ModulePathError) as error:
        resolve_module_path(root, "transformer.h.2.mlp")

    assert error.value.path == "transformer.h.2.mlp"
    assert error.value.segment == "2"
    assert error.value.resolved_prefix == "transformer.h"
    assert "registered module" in str(error.value)


def test_sequence_index_out_of_range_is_actionable() -> None:
    with pytest.raises(ModulePathError, match="index 3 is outside sequence range") as error:
        resolve_module_path({"layers": [object()]}, "layers.3")

    assert error.value.resolved_prefix == "layers"


def test_none_registered_module_fails_closed() -> None:
    root = ModuleNode(optional=None)

    with pytest.raises(ModulePathError, match="resolved to null"):
        resolve_module_path(root, "optional")
