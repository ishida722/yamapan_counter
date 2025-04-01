import cv2
import numpy as np
import pytest

from yamapan_counter.detector.point_detector import PointImageDetector, match_result


@pytest.fixture
def sample_image():
    # Create a dummy image for testing
    return cv2.imread("experiments/data/IMG_4183_sheet.png")
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.rectangle(image, (400, 400), (600, 600), (255, 255, 255), -1)
    return image


@pytest.fixture
def detector():
    return PointImageDetector()


def test_match_template(detector, sample_image):
    matches = detector.match_template(sample_image, n=1)
    assert len(matches) == 1
    assert isinstance(matches[0], match_result)
    assert matches[0].value > 0


def test_filter_match(detector, sample_image):
    matches = detector.match_template(sample_image, n=5)
    filtered_matches = detector.filter_match(matches)
    assert len(filtered_matches) <= len(matches)
    assert all(m.value > 0 for m in filtered_matches)


def test_get_match_images(detector, sample_image):
    matches = detector.match_template(sample_image, n=1)
    match_images = detector.get_match_images(sample_image, matches)
    assert len(match_images) == 1
    assert match_images[0].shape == (500, 500, 3)


def test_replace_images(detector, sample_image):
    matches = detector.match_template(sample_image, n=1)
    match_images = detector.get_match_images(sample_image, matches)
    replaced_image = detector.replace_images(sample_image, matches, match_images)
    assert replaced_image.shape == sample_image.shape


def test_highlight_points(detector, sample_image):
    matches = detector.match_template(sample_image, n=1)
    highlighted_image = detector.highlight_points(sample_image, matches)
    assert highlighted_image.shape == sample_image.shape


def test_binarize(detector, sample_image):
    binarized_image = detector.binarize(sample_image)
    assert binarized_image.shape[:2] == sample_image.shape[:2]
    assert binarized_image.dtype == np.uint8


def test_get_point_image(detector, sample_image):
    matches = detector.match_template(sample_image, n=1)
    match_images = detector.get_match_images(sample_image, matches)
    point_image = detector.get_point_image(match_images[0])
    assert point_image.shape == (500, 500)
    assert point_image.dtype == np.uint8
