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

st.title("リアルタイム顔検出アプリ")

frame_placeholder = st.empty()
start_button_pressed = st.button("再開")
stop_button_pressed = st.button("判定")

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
    # フレームをセッションに保存
    st.session_state["frame"] = frame

    if not ret:
        st.write("カメラから映像を取得できませんでした。")
        break
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
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

frame = st.session_state["frame"]
frame_placeholder.image(frame, channels="BGR")
# frameからガイド矩形(ターゲット)を切り取り
guid_pt1, guid_pt2 = st.session_state["guid_pt"]
target = frame[guid_pt1[1] : guid_pt2[1], guid_pt1[0] : guid_pt2[0]]
# ターゲットをリサイズ
target = cv2.resize(target, (2600, 3900))
# ポイント画像検出
with st.spinner("画像検出中..."):
    detector = PointImageDetector()
    ocr = PointImageOcr()
    point_image = detector.get_point_images(target)
    points = [ocr.read_point(im) for im in point_image]
st.write(f"{sum(points)}点")

cap.release()
