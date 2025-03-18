# %%

import random
from pathlib import Path

import cv2
import easyocr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from PIL import Image

# %%
image_paths = [p for p in Path("data").glob("*.png") if "sheet" not in p.name]
image_paths

# %%
imgs = [Image.open(p) for p in image_paths]
imgs = list(map(lambda img: img.resize((500, 500)), imgs))

# %%
# imgsをランダムに回転させて500x500にリサイズした画像を作成
rotated_imgs = []
for i in range(100):
    for img in imgs:
        rotated_imgs.append(img.rotate(random.randint(0, 360)).resize((500, 500)))
rotated_imgs[100]
# %%
# imgsを平均した画像を作成
avg_img = Image.new("RGB", rotated_imgs[0].size)
for img in imgs:
    avg_img = Image.blend(avg_img, img, 1 / len(imgs))
avg_img.save("data/avg_img.png")
avg_img

# %%
sheet_img = Image.open("data/IMG_4183_sheet.png")
sheet_img

# %%
# Convert images to grayscale
avg_img_gray = avg_img.convert("L")
sheet_img_gray = sheet_img.convert("L")

avg_img_gray_array = np.array(avg_img_gray)
sheet_img_gray_array = np.array(sheet_img_gray)

# テンプレートマッチングの実行
result = cv2.matchTemplate(
    sheet_img_gray_array, avg_img_gray_array, cv2.TM_CCOEFF_NORMED
)
fig = go.Figure()
fig.add_heatmap(z=result)

# %%
# resultの値が高い順に10個、x,y,値を取得
top10 = []
result = cv2.matchTemplate(
    sheet_img_gray_array, avg_img_gray_array, cv2.TM_CCOEFF_NORMED
)
# resultを250x250を左と上に追加するようにして0にする
result = np.pad(result, ((250, 250), (250, 250)))
for i in range(30):
    y, x = np.unravel_index(np.argmax(result), result.shape)
    # y, xからの500x500の範囲を0にする
    y1 = max(y - 250, 0)
    y2 = min(y + 250, result.shape[0])
    x1 = max(x - 250, 0)
    x2 = min(x + 250, result.shape[1])
    top10.append((x, y, result[y, x], x1, y1, x2, y2))
    result[y1:y2, x1:x2] = 0
print(top10)
fig = go.Figure()
fig.add_heatmap(z=result)

# %%

df_top = pl.DataFrame(top10, schema=["x", "y", "value", "x1", "y1", "x2", "y2"])
# %%
# 閾値を設定
th = (
    df_top.sort("value", descending=True)
    .with_columns(pl.col("value").diff().alias("diff"))
    .drop_nans()
    .sort("diff", descending=False)[0]["value"]
)
df_top_th = df_top.filter(pl.col("value") > th)
# sheet_img に　top10の座標を描画
sheet_img_draw = np.array(sheet_img).copy()
for x, y, _, x1, y1, x2, y2 in df_top_th.iter_rows():
    cv2.rectangle(sheet_img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
Image.fromarray(sheet_img_draw)

# %%
px.line(df_top, y="value")


# %%
# imを二値化して
def binarize(im):
    im_gray = im.convert("L")
    im_gray_array = np.array(im_gray)
    _, im_bin = cv2.threshold(
        im_gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return im_bin


def get_point_image(image_mat: np.ndarray):
    min_size = 5000
    max_size = 200000
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_mat)
    point_image = np.zeros_like(image_mat)
    for i in range(1, num_labels):
        l = stats[i, cv2.CC_STAT_LEFT]
        t = stats[i, cv2.CC_STAT_TOP]
        norm_stat = np.linalg.norm(np.array([l, t]) - np.array([250, 250]))
        norm_cent = np.linalg.norm(centroids[i] - np.array([250, 250]))
        if norm_stat > 200:
            continue
        if norm_cent > 200:
            continue
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            continue
        if stats[i, cv2.CC_STAT_AREA] > max_size:
            continue
        point_image[labels == i] = 255
    return point_image


def corretion_angle(image_mat: np.ndarray):
    # im_binから輪郭を抽出する
    contours, hierarchy = cv2.findContours(
        image_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 輪郭から直線を求める
    line = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    line = line.flatten()
    # lineの角度を求める
    angle = -np.arctan2(line[1], line[0]) * 180 / np.pi
    # # lineを描画
    # im_bin = cv2.cvtColor(im_bin, cv2.COLOR_GRAY2BGR)
    # # line = np.intp(line)
    # pt1 = (int(line[2]), int(line[3]))
    # # pt2 = (line[2] + line[0] * 100, line[3] + line[1] * 100)
    # pt2 = np.intp((pt1[0] + line[0] * 1000, pt1[1] + line[1] * 1000))
    # print(pt2, line[0])
    # im_bin = cv2.line(im_bin, pt1, pt2, (0, 0, 255), 2)

    # 角度補正
    image = Image.fromarray(image_mat)
    # angleが90度になるように回転
    if angle > 90:
        new_angle = angle - 90
    elif angle < 0:
        new_angle = -angle + 90
    else:
        new_angle = 90 - angle
    image = image.rotate(new_angle)
    return_image_mat = np.array(image)
    return return_image_mat


im = imgs[11]
im = binarize(im)
im = get_point_image(im)
# im = corretion_angle(im)
Image.fromarray(im)

# %%

# easyocrで文字認識

reader = easyocr.Reader(["en"], gpu=False)
# %%


def check_ocr_results(ocr_results) -> bool:
    for result in ocr_results:
        print(result[-1])
        if result[-1] < 0.95:
            return False
    return True


def rotate_image(image_mat: np.ndarray, angle: float):
    image = Image.fromarray(image_mat)
    image = image.rotate(angle)
    return np.array(image)


def read_point(image_mat: np.ndarray):
    # return reader.readtext(image_mat, allowlist="0123456789")
    return reader.readtext(image_mat)


def get_point(image_mat: np.ndarray):
    ocr_results = read_point(image_mat)
    print(ocr_results)
    if check_ocr_results(ocr_results):
        return [result[-2] for result in ocr_results]
    for angle in range(0, 360, 10):
        image = rotate_image(image_mat, angle)
        ocr_results = read_point(image)
        print(ocr_results)
        if check_ocr_results(ocr_results):
            return [result[-2] for result in ocr_results]
    return None


im = imgs[9]
im = binarize(im)
im = get_point_image(im)
print(get_point(im))
Image.fromarray(im)

# %%
check_ocr_results([])

#%%
# img = cv2.imread("data/IMG_4182_all.png")
img = cv2.imread("data/IMG_4183_sheet.png")
# QRコード検出器を初期化
qr_detector = cv2.QRCodeDetectorAruco()

# QRコードを検出（デコードはしない）
retval, points = qr_detector.detect(img)
print(retval)

points = points[0].astype(int)

# 検出したQRコードの領域を矩形で描画
for i in range(len(points)):
    pt1 = tuple(points[i])
    pt2 = tuple(points[(i + 1) % len(points)])
    cv2.line(img, pt1, pt2, (255, 0, 0), 3)

Image.fromarray(img)