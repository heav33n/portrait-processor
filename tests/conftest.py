import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from portrait_processor.document import Document, Face


@pytest.fixture
def portrait(tmp_path):
    rgb = np.full((600, 800, 3), (190, 210, 235), dtype=np.uint8)
    rgb[40:560, 160:640] = (220, 150, 120)
    rgb[300:560, 160:400] = (250, 250, 250)
    rgb[300:560, 400:640] = (40, 100, 200)
    path = tmp_path / "portrait.png"
    Image.fromarray(rgb).save(path)
    doc = Document.load(path)
    alpha = np.zeros((600, 800), dtype=np.uint8)
    alpha[40:560, 160:640] = 255
    doc.apply_analysis(alpha, [Face((399.5, 220), (330, 160), (470, 160))])
    return doc
