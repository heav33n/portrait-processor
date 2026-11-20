from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

WIDTH: int
HEIGHT: int

class RenderResult(TypedDict):
    rgb: NDArray[np.uint8]
    before_rgb: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    before_alpha: NDArray[np.uint8]
    transform: list[float]
    inverse: list[float]
    scale: float
    removed_fraction: float

def render(
    rgb: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    nose: tuple[float, float],
    angle_degrees: float = 0.0,
    headroom: int = 180,
    side_margin: int = 64,
    bottom_margin: int = 0,
    symmetry: bool = True,
    horizontal_offset: int = 0,
) -> RenderResult: ...
def clean_specks(
    alpha: NDArray[np.uint8],
    min_area: int = 24,
    anchor: tuple[float, float] | None = None,
) -> NDArray[np.uint8]: ...
