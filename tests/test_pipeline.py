from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from portrait_processor import _native
from portrait_processor.document import (
    Document,
    Face,
    export_stem,
    render_document,
    save_image,
    save_png,
    validate_windows_name,
)


def test_import_only_position_survives_draft_reload(portrait, tmp_path):
    doc = Document.load(portrait.path)
    doc.manual_placement = True
    doc.settings.position_x = 28
    doc.settings.position_y = 14
    doc.output_name = "Custom name"
    doc.export_format = "JPEG"
    doc.export_quality = 87
    doc.changed()
    expected = render_document(doc)
    draft = tmp_path / "manual.npz"
    doc.stash(draft)
    restored = Document.from_stash(portrait.path, draft)
    assert restored.ready and restored.manual_placement and not restored.analyzed
    assert restored.nose is None
    assert restored.output_name == "Custom name" and restored.export_format == "JPEG"
    assert restored.export_quality == 87
    np.testing.assert_array_equal(render_document(restored)["rgb"], expected["rgb"])


def test_moving_down_covers_a_tilted_lower_edge_without_resizing(portrait):
    portrait.settings.angle_offset = 6
    before = render_document(portrait)
    portrait.settings.position_y = 30
    after = render_document(portrait)
    assert after["scale"] == before["scale"]
    np.testing.assert_array_equal(after["alpha"][-1], before["alpha"][-31])
    np.testing.assert_array_equal(after["rgb"][-1], before["rgb"][-31])


def test_exact_canvas_symmetric_alpha_original_colors_and_centered_nose(portrait):
    portrait.settings.strict_symmetry = True
    result = render_document(portrait)
    assert result["rgb"].shape == (1048, 948, 3)
    assert np.array_equal(result["alpha"], result["alpha"][:, ::-1])
    assert np.all(result["alpha"] <= result["before_alpha"])
    assert np.any(result["rgb"] != result["rgb"][:, ::-1])
    assert np.all(result["rgb"][:180] == 255)
    assert np.any(result["alpha"][-1])
    t = result["transform"]
    nx, ny = portrait.nose
    assert t[0] * nx + t[1] * ny + t[2] == pytest.approx(473.5)
    assert result["scale"] == 1


def test_head_limit_shrinks_subject_while_retaining_bottom_anchor(portrait):
    first = render_document(portrait)
    portrait.settings.headroom = 700
    second = render_document(portrait)
    assert first["scale"] == 1
    assert second["scale"] < 1
    ys, _ = np.nonzero(second["alpha"])
    assert ys.min() >= 700
    assert ys.max() == 1047


def test_small_portrait_is_bottom_aligned_without_enlarging(portrait):
    first = render_document(portrait)
    portrait.settings.headroom = 100
    second = render_document(portrait)
    assert first["scale"] == second["scale"] == 1
    np.testing.assert_array_equal(first["alpha"], second["alpha"])


@pytest.mark.parametrize("angle", [0, 9.5, -17])
@pytest.mark.parametrize("strict", [False, True])
def test_top_sides_and_bottom_constraints_survive_rotation_and_trimming(portrait, angle, strict):
    portrait.settings.headroom = 300
    portrait.settings.side_margin = 180
    portrait.settings.bottom_margin = 35
    portrait.settings.angle_offset = angle
    portrait.settings.strict_symmetry = strict
    result = render_document(portrait)
    ys, xs = np.nonzero(result["alpha"])
    assert ys.min() >= 300
    assert ys.max() == 1048 - 35 - 1
    assert xs.min() >= 180
    assert xs.max() < 948 - 180
    if strict:
        assert np.array_equal(result["alpha"], result["alpha"][:, ::-1])
    else:
        np.testing.assert_array_equal(result["alpha"], result["before_alpha"])
    assert np.all(result["alpha"] <= result["before_alpha"])
    t = result["transform"]
    inv = result["inverse"]
    x, y = portrait.nose
    px, py = t[0] * x + t[1] * y + t[2], t[3] * x + t[4] * y + t[5]
    assert px == pytest.approx(473.5)
    assert inv[0] * px + inv[1] * py + inv[2] == pytest.approx(x)
    assert inv[3] * px + inv[4] * py + inv[5] == pytest.approx(y)


def test_natural_outline_retains_asymmetric_hair(portrait):
    portrait.alpha[100:200, 90:160] = 255
    assert not portrait.settings.strict_symmetry
    natural = render_document(portrait)
    np.testing.assert_array_equal(natural["alpha"], natural["before_alpha"])
    assert natural["removed_fraction"] == 0
    portrait.settings.strict_symmetry = True
    strict = render_document(portrait)
    assert strict["removed_fraction"] > 0
    assert np.array_equal(strict["alpha"], strict["alpha"][:, ::-1])
    for result, expected in [(natural, 255), (strict, 0)]:
        t = result["transform"]
        x, y = round(t[0] * 120 + t[1] * 150 + t[2]), round(t[3] * 120 + t[4] * 150 + t[5])
        assert result["alpha"][y, x] == expected


@pytest.mark.parametrize("bottom", [0, 40])
@pytest.mark.parametrize("color", [(80, 110, 145), (255, 255, 255)])
def test_soft_bottom_edge_does_not_leave_a_white_strip(bottom, color):
    rgb = np.full((400, 320, 3), color, dtype=np.uint8)
    alpha = np.zeros((400, 320), dtype=np.uint8)
    alpha[30:350, 60:260] = 255
    for y, opacity in enumerate([240, 180, 100, 25, 5], 350):
        alpha[y, 60:260] = opacity
    result = _native.render(rgb, alpha, (159.5, 150), bottom_margin=bottom, symmetry=False)
    row = 1047 - bottom
    assert np.count_nonzero(result["alpha"][row] >= 250) >= 190
    np.testing.assert_array_equal(result["rgb"][row, 474], color)
    assert not np.any(result["alpha"][row + 1 :])


def test_tilted_eyes_become_level(portrait):
    portrait.eye_b = (470, 200)
    result = render_document(portrait)
    t = result["transform"]
    a, b = portrait.eye_a, portrait.eye_b
    assert t[3] * a[0] + t[4] * a[1] + t[5] == pytest.approx(t[3] * b[0] + t[4] * b[1] + t[5])


def test_large_image_shrinks_and_fits():
    rgb = np.full((1800, 1600, 3), 255, dtype=np.uint8)
    alpha = np.full((1800, 1600), 255, dtype=np.uint8)
    result = _native.render(rgb, alpha, (799.5, 300))
    assert 0 < result["scale"] < 1
    ys, xs = np.nonzero(result["alpha"])
    assert xs.min() >= 64 and xs.max() < 948 - 64
    assert ys.min() >= 180 and ys.max() == 1047
    # White foreground pixels remain foreground even though the canvas is also white.
    assert result["alpha"].max() == 255


def test_premultiplied_sampling_does_not_leak_hidden_blue():
    rgb = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)
    alpha = np.zeros((200, 200), dtype=np.uint8)
    alpha[20:180, 20:180] = 255
    rgb[20:180, 20:180] = (255, 0, 0)
    result = _native.render(rgb, alpha, (99.5, 90), 17)
    edges = (result["alpha"] > 0) & (result["alpha"] < 255)
    assert np.count_nonzero(edges) > 100
    assert np.all(result["rgb"][edges, 0] == 255)
    assert np.array_equal(result["rgb"][edges, 1], result["rgb"][edges, 2])


@pytest.mark.parametrize("nose,headroom", [((float("nan"), 3), 60), ((-1, 3), 60), ((3, 3), 1048)])
def test_invalid_placement_is_rejected(nose, headroom):
    with pytest.raises(ValueError):
        _native.render(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8),
            nose,
            headroom=headroom,
        )


def test_empty_mask_and_channel_mismatch_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        _native.render(
            np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), (3, 3)
        )
    with pytest.raises(ValueError, match="dimensions"):
        _native.render(
            np.zeros((10, 10, 4), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8), (3, 3)
        )


def test_speck_cleanup_preserves_anchor_and_big_components():
    alpha = np.zeros((50, 50), dtype=np.uint8)
    alpha[1, 1] = 255
    alpha[1, 45] = 255
    alpha[25:45, 25:45] = 255
    cleaned = _native.clean_specks(alpha, 4, (1, 1))
    assert cleaned[1, 1] == 255
    assert cleaned[1, 45] == 0
    assert np.all(cleaned[25:45, 25:45] == 255)


def test_restore_can_recover_ai_removed_pixels_and_undo_redo(portrait):
    before = portrait.alpha.copy()
    portrait.paint((100, 100), 30, True)
    portrait.record_mask(before)
    assert portrait.alpha[100, 100] == 255
    portrait.undo()
    np.testing.assert_array_equal(portrait.alpha, before)
    portrait.redo()
    assert portrait.alpha[100, 100] == 255
    before_erase = portrait.alpha.copy()
    portrait.paint((400, 300), 25, False)
    portrait.record_mask(before_erase)
    assert portrait.alpha[300, 400] == 0
    portrait.undo()
    assert portrait.alpha[300, 400] == 255


def test_restore_respects_input_transparency(portrait):
    portrait.original_alpha[90:110, 90:110] = 0
    portrait.paint((100, 100), 10, True)
    assert portrait.alpha[100, 100] == 0


def test_multiple_faces_need_selection(portrait):
    faces = [Face((200, 200), (180, 170), (220, 170)), Face((600, 200), (580, 170), (620, 170))]
    portrait.apply_analysis(portrait.alpha, faces)
    assert not portrait.ready
    portrait.choose_face(1)
    assert portrait.ready and portrait.nose == faces[1].nose


def test_export_is_exact_rgb_and_collision_safe(portrait, tmp_path):
    rgb = render_document(portrait)["rgb"]
    directory = tmp_path / "processed_photos"
    first = save_png(rgb, directory, "portrait")
    second = save_png(rgb, directory, "portrait")
    assert first.name == "portrait.png"
    assert second.name == "portrait_2.png"
    with Image.open(first) as exported:
        assert exported.size == (948, 1048)
        assert exported.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(exported), rgb)
    assert set(directory.iterdir()) == {first, second}


def test_export_failure_leaves_no_partial_file(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="Disk full"):
        save_png(np.zeros((1048, 948, 3), dtype=np.uint8), tmp_path, "failed")
    assert list(tmp_path.iterdir()) == []


def test_exif_orientation_is_applied(tmp_path):
    image = Image.new("RGB", (80, 40))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    assert Document.load(path).rgb.shape[:2] == (80, 40)


def test_source_photo_is_not_modified(portrait):
    original = Path(portrait.path).read_bytes()
    render_document(portrait)
    assert portrait.path.read_bytes() == original


def test_session_draft_preserves_mask_landmarks_settings_and_history(portrait, tmp_path):
    original_mask = portrait.alpha.copy()
    portrait.paint((400, 300), 30, False)
    portrait.record_mask(original_mask)
    portrait.settings.headroom = 140
    portrait.settings.strict_symmetry = True
    portrait.settings.horizontal_offset = 37
    portrait.export_format = "JPEG"
    portrait.export_quality = 87
    portrait.output_name = "Custom portrait"
    draft = tmp_path / "session.npz"
    portrait.stash(draft)
    restored = Document.from_stash(portrait.path, draft)
    assert restored.ready
    assert restored.nose == portrait.nose
    assert restored.faces == portrait.faces
    assert restored.settings.headroom == 140
    assert restored.settings.strict_symmetry
    assert restored.settings.horizontal_offset == 37
    assert restored.export_format == "JPEG" and restored.export_quality == 87
    assert restored.output_name == "Custom portrait"
    assert restored.dirty
    np.testing.assert_array_equal(restored.alpha, portrait.alpha)
    restored.undo()
    np.testing.assert_array_equal(restored.alpha, original_mask)


def test_session_draft_rejects_changed_source_dimensions(portrait, tmp_path):
    draft = tmp_path / "session.npz"
    portrait.stash(draft)
    Image.new("RGB", (10, 10)).save(portrait.path)
    with pytest.raises(ValueError, match="dimensions changed"):
        Document.from_stash(portrait.path, draft)


@pytest.mark.parametrize(
    "value", ["", " ", ".png", "..", "../outside", "folder/name", "folder\\name", "bad\x00name"]
)
def test_export_name_must_be_a_plain_nonempty_filename(value):
    with pytest.raises(ValueError):
        export_stem(value)


def test_custom_name_with_png_extension_does_not_get_a_second_extension(portrait, tmp_path):
    result = render_document(portrait)
    path = save_png(result["rgb"], tmp_path / "custom", "  Mi retrato.PNG  ")
    assert path.name == "Mi retrato.png"


@pytest.mark.parametrize(
    "image_format,extension",
    [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp"), ("TIFF", "tiff"), ("BMP", "bmp")],
)
def test_export_formats_and_collisions(portrait, tmp_path, image_format, extension):
    rgb = render_document(portrait)["rgb"]
    first = save_image(rgb, tmp_path, "My portrait.png", image_format, 92)
    second = save_image(rgb, tmp_path, "My portrait", image_format, 92)
    assert first.name == f"My portrait.{extension}"
    assert second.name == f"My portrait_2.{extension}"
    with Image.open(first) as photo:
        assert photo.size == (948, 1048) and photo.format == image_format
        if image_format in {"PNG", "TIFF", "BMP"}:
            np.testing.assert_array_equal(np.asarray(photo), rgb)


def test_jpeg_quality_controls_compression(tmp_path):
    rgb = np.random.default_rng(17).integers(0, 256, (1048, 948, 3), dtype=np.uint8)
    low = save_image(rgb, tmp_path, "low", "JPEG", 15)
    high = save_image(rgb, tmp_path, "high", "JPEG", 95)
    assert high.stat().st_size > low.stat().st_size


@pytest.mark.parametrize(
    "name", ["CON", "NUL", "COM1", "LPT2.photo", "COM¹", "photo?", "photo:", "photo."]
)
def test_windows_reserved_filenames_are_rejected(name):
    with pytest.raises(ValueError):
        validate_windows_name(name)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("offset", [-70, 65])
def test_manual_horizontal_position_keeps_frame_constraints(portrait, strict, offset):
    portrait.settings.horizontal_offset = offset
    portrait.settings.strict_symmetry = strict
    result = render_document(portrait)
    t = result["transform"]
    x, y = portrait.nose
    assert t[0] * x + t[1] * y + t[2] == pytest.approx(473.5 + offset)
    ys, xs = np.nonzero(result["alpha"])
    assert ys.min() >= 180 and ys.max() == 1047
    assert xs.min() >= 64 and xs.max() < 884
    if strict:
        reflected = 947 + 2 * offset - np.arange(948)
        inside = (reflected >= 0) & (reflected < 948)
        np.testing.assert_array_equal(
            result["alpha"][:, inside], result["alpha"][:, reflected[inside]]
        )
        assert not np.any(result["alpha"][:, ~inside])
