import easyocr
import numpy as np
from PIL import Image


class PointImageOcr:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)

    def _read_point(self, image_mat: np.ndarray):
        return self.reader.readtext(image_mat, allowlist="0123456789")

    def check_ocr_results(self, ocr_results: list) -> bool:
        if not ocr_results:
            return False
        for result in ocr_results:
            print(result[-1])
            if result[-1] < 0.95:
                return False
        return True

    def convert_str_to_point(self, point_str: str) -> float:
        if point_str == "1":
            return 1
        if point_str == "15":
            return 1.5
        if point_str == "2":
            return 2
        if point_str == "25":
            return 2.5
        if point_str == "3":
            return 3
        return 0

    def read_point(self, image_mat: np.ndarray) -> float:
        # 様々な角度でOCRを試す
        for angle in range(0, 360, 10):
            # 画像を回転
            image = self.rotate_image(image_mat, angle)
            # 画像から数字を読み取る
            ocr_results = self._read_point(image)
            # 読み取り結果が正しいか確認
            if self.check_ocr_results(ocr_results):
                # 正しい結果が得られた場合は数字のリストを返す
                point_str = "".join([result[-2] for result in ocr_results])
                return self.convert_str_to_point(point_str)
        return None

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return image
        image = Image.fromarray(image)
        image = image.rotate(angle)
        return np.array(image)
