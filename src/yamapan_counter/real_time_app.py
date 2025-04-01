import cv2
import streamlit as st

from yamapan_counter.detector.point_detector import PointImageDetector
from yamapan_counter.ocr.point_image_ocr import PointImageOcr

# カメラをキャプチャ（カメラIDは0をデフォルトに設定）
cap = cv2.VideoCapture(0)

# 現在のフレームを保持するための変数
if "frame" not in st.session_state:
    st.session_state["frame"] = None

if "guid_pt" not in st.session_state:
    st.session_state["guid_pt"] = None

st.title("ヤマザキ春のパン祭り 点数カウンター")
st.markdown(""" 
# 使い方
1. 画像の四角内にポイントシートを合わせてください。
2. 「判定」ボタンを押してください。
3. しばらく待つと処理後の画像と総ポイントが表示されます。

デモボタンを押すと、デモ画像で処理を行います。
""")

cols = st.columns(2)
with cols[0]:
    frame_placeholder = st.empty()
with cols[1]:
    after_placeholder = st.empty()

start_button_pressed = st.button("再開")
stop_button_pressed = st.button("判定")
demo_button_pressed = st.button("デモ")


def is_playing_camera():
    if stop_button_pressed:
        return False
    if demo_button_pressed:
        return False
    return True


while is_playing_camera():
    ret, frame = cap.read()
    # フレームをセッションに保存
    st.session_state["frame"] = frame

    if not ret:
        st.write("カメラから映像を取得できませんでした。")
        break
    # フレームサイズ取得
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # フレーム内ガイド矩形サイズを計算
    guid_h = int(frame_h * 0.8)
    guid_w = int(guid_h * 0.66)
    # frameにガイド矩形を描画
    c_h = int(frame_h * 0.1)
    c_w = int(frame_w / 2) - int(guid_w / 2)
    guid_pt1 = (c_w, c_h)
    guid_pt2 = (c_w + guid_w, c_h + guid_h)
    st.session_state["guid_pt"] = (guid_pt1, guid_pt2)
    frame = cv2.rectangle(frame, guid_pt1, guid_pt2, (255, 0, 0), 3)
    # 表示
    frame_placeholder.image(frame, channels="BGR")

# デモボタンが押されている場合、デモ画像で処理する
if demo_button_pressed:
    frame = cv2.imread("experiments/data/IMG_4183_sheet.png")
    target = frame
else:
    frame = st.session_state["frame"]
    # frameからガイド矩形(ターゲット)を切り取り
    guid_pt1, guid_pt2 = st.session_state["guid_pt"]
    target = frame[guid_pt1[1] : guid_pt2[1], guid_pt1[0] : guid_pt2[0]]
    # ターゲットをリサイズ
    target = cv2.resize(target, (2600, 3900))

frame_placeholder.image(frame, channels="BGR")
# ポイント画像検出
with st.spinner("画像検出中..."):
    detector = PointImageDetector()
    ocr = PointImageOcr()
    # ポイントシールを検出
    result = detector.get_point_images(target)
    # 表示を更新
    after_placeholder.image(result.replaced_image, channels="BGR")
    # シール画像からポイント読み取り、算出
    points = [ocr.read_point(im) for im in result.images]

st.write(f"{sum(points)}点")

cap.release()
