import cv2
import streamlit as st

# カメラをキャプチャ（カメラIDは0をデフォルトに設定）
cap = cv2.VideoCapture(0)

st.title("リアルタイム顔検出アプリ")

frame_placeholder = st.empty()
stop_button_pressed = st.button("停止")

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()
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
    frame = cv2.rectangle(
        frame, (c_w, c_h), (c_w + guid_w, c_h + guid_h), (255, 0, 0), 3
    )
    # 表示
    frame_placeholder.image(frame, channels="BGR")
    # frameからガイド矩形(ターゲット)を切り取り
    # 矩形をリサイズ
    # 

cap.release()
