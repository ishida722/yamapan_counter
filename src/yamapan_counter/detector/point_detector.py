from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np
import polars as pl


@dataclass
class match_result:
    x: int
    y: int
    value: float
    x1: int
    y1: int
    x2: int
    y2: int

    @staticmethod
    def to_polars(match: list[match_result]) -> pl.DataFrame:
        return pl.DataFrame(
            [(m.x, m.y, m.value, m.x1, m.y1, m.x2, m.y2) for m in match],
            schema=["x", "y", "value", "x1", "y1", "x2", "y2"],
            orient="row",
        )

    @staticmethod
    def from_polars(df: pl.DataFrame) -> list[match_result]:
        return [match_result(*row) for row in df.iter_rows()]


@dataclass
class PointImageDetectorResult:
    images: list[np.ndarray] = field(
        metadata={"description": "検出されたポイント画像のリスト"}
    )
    rect_point_image: np.ndarray = field(
        metadata={"description": "ポイントを強調表示した画像"}
    )
    replaced_image: np.ndarray = field(
        metadata={"description": "元画像を抽出したポイント画像で置き換えた画像"}
    )


class PointImageDetector:
    def __init__(self):
        template_image: np.ndarray = cv2.imread(
            "src/yamapan_counter/template/avg_img.png"
        )
        template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        self.template_image: Final[np.ndarray] = template_image

    def get_point_images(
        self,
        image: np.ndarray,
        n: int = 30,
    ) -> PointImageDetectorResult:
        # テンプレートマッチング
        match = self.match_template(image, n)
        # マッチング結果のフィルタリング
        match = self.filter_match(match)
        # マッチング結果から画像を取得
        images = self.get_match_images(image, match)
        # ポイントを強調表示した画像を作成
        rect_point_image = self.highlight_points(image, match)
        # 画像からポイント画像を取得
        images = [self.get_point_image(im) for im in images]
        # 元画像を抽出したポイント画像で置き換えた画像を作成
        replaced_image = self.replace_images(image, match, images)
        return PointImageDetectorResult(
            images=images,
            rect_point_image=rect_point_image,
            replaced_image=replaced_image,
        )

    def match_template(self, image: np.ndarray, n: int = 30) -> list[match_result]:
        match = []
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(
            image_gray,
            self.template_image,
            cv2.TM_CCOEFF_NORMED,
        )
        # resultを250x250を左と上に追加するようにして0にする
        result = np.pad(result, ((250, 250), (250, 250)))
        for _ in range(n):
            y, x = np.unravel_index(np.argmax(result), result.shape)
            # y, xからの500x500の範囲を0にする
            y1 = max(y - 250, 0)
            y2 = min(y + 250, result.shape[0])
            x1 = max(x - 250, 0)
            x2 = min(x + 250, result.shape[1])
            match.append(match_result(x, y, result[y, x], x1, y1, x2, y2))
            # 取得したエリアの値を0にする
            result[y1:y2, x1:x2] = 0
        return match

    def filter_match(
        self, match: list[match_result], threshold: float | None = None
    ) -> list[match_result]:
        if threshold is not None:
            return [m for m in match if m.value > threshold]

        df = match_result.to_polars(match)
        th = (
            df.sort("value", descending=True)
            .with_columns(pl.col("value").diff().alias("diff"))
            .drop_nans()
            .sort("diff", descending=False)[0]["value"]
        )
        ret_df = df.filter(pl.col("value") > th)
        return match_result.from_polars(ret_df)

    def get_match_images(
        self, image: np.ndarray, match: list[match_result]
    ) -> list[np.ndarray]:
        return [image[m.y1 : m.y2, m.x1 : m.x2] for m in match]

    # 元画像を抽出したポイント画像で置き換えた画像を作成
    def replace_images(
        self,
        image: np.ndarray,
        match: list[match_result],
        point_images: list[np.ndarray],
    ) -> np.ndarray:
        new_image = image.copy()
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        for m, point_image in zip(match, point_images):
            new_image[m.y1 : m.y2, m.x1 : m.x2] = point_image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        return new_image

    # ポイント部分を強調したあたらしい画像を作成
    def highlight_points(
        self,
        image: np.ndarray,
        match: list[match_result],
    ) -> np.ndarray:
        new_image = image.copy()
        for m in match:
            cv2.rectangle(
                new_image,
                (m.x1, m.y1),
                (m.x2, m.y2),
                (0, 255, 0),
                2,
            )
        return new_image

    def binarize(self, image: np.ndarray) -> np.ndarray:
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_gray_array = np.array(im_gray)
        _, im_bin = cv2.threshold(
            im_gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return im_bin

    def get_point_image(
        self, image: np.ndarray, min_size: int = 5000, max_size: int = 200000
    ) -> np.ndarray:
        image = self.binarize(image)
        # オブジェクトをラベリング
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        # 返り値用の画像を生成
        point_image = np.zeros_like(image)
        # オブジェクトごとに処理
        for i in range(1, num_labels):
            # オブジェクトの左上座標を取得
            l = stats[i, cv2.CC_STAT_LEFT]
            t = stats[i, cv2.CC_STAT_TOP]
            # オブジェクトの中心座標と画像中心座標の距離を計算
            norm_stat = np.linalg.norm(np.array([l, t]) - np.array([250, 250]))
            norm_cent = np.linalg.norm(centroids[i] - np.array([250, 250]))
            # 外周部にあるオブジェクトは除外
            if norm_stat > 200:
                continue
            if norm_cent > 200:
                continue
            # 面積が一定範囲外のオブジェクトは除外
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                continue
            if stats[i, cv2.CC_STAT_AREA] > max_size:
                continue
            point_image[labels == i] = 255
        return point_image
