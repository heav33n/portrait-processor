import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from portrait_processor.models import expand_model, pack_model, prepare_model


def tiny_model(path, weight):
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(weight, "w")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[
        0
    ]


def test_float16_pack_halves_weights_and_expands_to_matching_float32(tmp_path):
    big = np.random.default_rng(2).normal(size=(3, 2)).astype(np.float32)
    original, packed, expanded = (tmp_path / n for n in ("o.onnx", "p.onnx", "e.onnx"))
    tiny_model(original, big)
    pack_model(original, packed)
    stored = onnx.load(packed).graph.initializer[0]
    assert stored.data_type == TensorProto.FLOAT16
    expand_model(packed, expanded)
    restored = onnx.load(expanded).graph.initializer[0]
    assert restored.data_type == TensorProto.FLOAT
    x = np.ones((1, 3), np.float32)
    np.testing.assert_allclose(run(expanded, x), run(original, x), atol=1e-2)


def test_values_outside_float16_range_stay_float32(tmp_path):
    weight = np.array([[1e6, 1.0], [2.0, 3.0], [4.0, 5.0]], np.float32)
    original, packed, expanded = (tmp_path / n for n in ("o.onnx", "p.onnx", "e.onnx"))
    tiny_model(original, weight)
    pack_model(original, packed)
    assert onnx.load(packed).graph.initializer[0].data_type == TensorProto.FLOAT
    expand_model(packed, expanded)
    x = np.ones((1, 3), np.float32)
    np.testing.assert_array_equal(run(expanded, x), run(original, x))


def test_prepare_model_expands_bundle_once_into_rembg_layout(tmp_path):
    bundle, home = tmp_path / "bundle", tmp_path / "home"
    bundle.mkdir()
    tiny_model(tmp_path / "o.onnx", np.ones((3, 2), np.float32))
    pack_model(tmp_path / "o.onnx", bundle / "u2net_human_seg.fp16.onnx")
    messages = []
    prepare_model("u2net_human_seg", home, messages.append, bundle)
    target = home / "models" / "u2net_human_seg" / "u2net_human_seg.onnx"
    assert target.is_file() and messages
    stamp = target.stat().st_mtime_ns
    prepare_model("u2net_human_seg", home, messages.append, bundle)
    assert target.stat().st_mtime_ns == stamp
    assert not list(target.parent.glob("*.tmp"))


@pytest.mark.parametrize("content", [None, b"not a model"])
def test_missing_or_corrupt_bundle_falls_back_to_download(tmp_path, content):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    if content is not None:
        (bundle / "isnet-general-use.fp16.onnx").write_bytes(content)
    prepare_model("isnet-general-use", tmp_path / "home", lambda _: None, bundle)
    folder = tmp_path / "home" / "models" / "isnet-general-use"
    assert not folder.exists() or not any(folder.iterdir())


def test_antivirus_lock_on_the_new_file_is_retried(tmp_path, monkeypatch):
    # Review finding #4: Windows Defender briefly holds freshly written files open.
    import os

    from portrait_processor import models

    bundle, home = tmp_path / "bundle", tmp_path / "home"
    bundle.mkdir()
    tiny_model(tmp_path / "o.onnx", np.ones((3, 2), np.float32))
    pack_model(tmp_path / "o.onnx", bundle / "u2net_human_seg.fp16.onnx")
    real_replace, attempts = os.replace, []

    def flaky(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(32, "The process cannot access the file")
        real_replace(source, target)

    monkeypatch.setattr(models.os, "replace", flaky)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    prepare_model("u2net_human_seg", home, lambda _: None, bundle)
    assert (home / "models" / "u2net_human_seg" / "u2net_human_seg.onnx").is_file()
    assert len(attempts) == 3


def test_persistent_lock_falls_back_without_raising(tmp_path, monkeypatch):
    from portrait_processor import models

    bundle, home = tmp_path / "bundle", tmp_path / "home"
    bundle.mkdir()
    tiny_model(tmp_path / "o.onnx", np.ones((3, 2), np.float32))
    pack_model(tmp_path / "o.onnx", bundle / "u2net_human_seg.fp16.onnx")

    def locked(*_, **__):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(models.os, "replace", locked)
    monkeypatch.setattr(models.Path, "unlink", locked)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    prepare_model("u2net_human_seg", home, lambda _: None, bundle)
    assert not (home / "models" / "u2net_human_seg" / "u2net_human_seg.onnx").exists()
