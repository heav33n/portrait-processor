import numpy as np
from portrait_processor.ai import InferenceEngine, refine_edges
from portrait_processor.inference_process import InferenceProcess


def step_scene(size=1600, blur=8):
    # A 320 px mask upscaled to 1600 px leaves a ramp of roughly this width.
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:, size // 2 :] = (200, 170, 150)
    x = np.arange(size, dtype=np.float32)
    ramp = np.clip((x - (size // 2 - blur)) / (2 * blur), 0, 1)
    alpha = np.tile((ramp * 255).round().astype(np.uint8), (size, 1))
    return rgb, alpha


def edge_width(alpha):
    row = alpha[alpha.shape[0] // 2].astype(int)
    return int(((row > 25) & (row < 230)).sum())


def test_refinement_snaps_a_blurry_mask_to_the_photo_edge():
    rgb, alpha = step_scene()
    refined = refine_edges(rgb, alpha)
    assert refined.dtype == np.uint8 and refined.shape == alpha.shape
    assert edge_width(refined) <= 4 < edge_width(alpha)
    row = refined[800].astype(int)
    assert row[795] < 30 and row[804] > 225


def test_refinement_keeps_solid_areas_untouched():
    rgb, alpha = step_scene()
    refined = refine_edges(rgb, alpha)
    solid = (alpha == 0) | (alpha == 255)
    far = np.zeros_like(solid)
    far[:, :700] = far[:, 900:] = True
    np.testing.assert_array_equal(refined[far & solid], alpha[far & solid])


def test_engine_levels_choose_model_and_refinement():
    assert InferenceEngine(1).model_name == "u2net_human_seg"
    assert InferenceEngine(2).model_name == "u2net_human_seg"
    assert InferenceEngine(4).model_name == "birefnet-general-lite"
    process = InferenceProcess(level=3)
    assert process.level == 3 and process.model_name == "isnet-general-use"
