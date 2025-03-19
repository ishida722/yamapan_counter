from collections import namedtuple

import cv2
import pytest

from yamapan_counter.detector.point_detector import PointImageDetector
from yamapan_counter.ocr.point_image_ocr import PointImageOcr

TestImage = namedtuple("TestImage", ["path", "point"])

test_images = [
    TestImage("IMG_4183_2.png", 1.5),
    TestImage("IMG_4183_3.png", 2),
    TestImage("IMG_4183_4.png", 1.5),
    TestImage("IMG_4183_5.png", 1.5),
    TestImage("IMG_4183_6.png", 3),
    TestImage("IMG_4183_7.png", 1.5),
    TestImage("IMG_4183_8.png", 1.5),
    TestImage("IMG_4183_9.png", 2.5),
    TestImage("IMG_4183_10.png", 1.5),
    TestImage("IMG_4183_11.png", 2.5),
    TestImage("IMG_4183_12.png", 1),
    TestImage("IMG_4183_13.png", 1),
    TestImage("IMG_4183_14.png", 1.5),
]


@pytest.mark.parametrize("test_image", test_images)
def test_ocr(test_image: TestImage):
    detector = PointImageDetector()
    ocr = PointImageOcr()
    print(f"Testing {test_image.path}")
    im = cv2.imread(f"experiments/data/{test_image.path}")
    im = detector.get_point_image(im)
    point = ocr.read_point(im)
    assert point == test_image.point
