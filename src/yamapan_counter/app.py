import re

import easyocr
import numpy as np
import streamlit as st
from PIL import Image

st.title("画像から数字の合計を算出するアプリ (EasyOCR)")

st.markdown("---")
# iPhoneの場合は、カメラ入力が利用可能
image_file = st.camera_input("カメラで写真を撮る")
if image_file is None:
    image_file = st.file_uploader(
        "画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"]
    )

if image_file is not None:
    try:
        # 画像ファイルを読み込み表示
        image = Image.open(image_file)
        st.image(image, caption="入力画像", use_column_width=True)

        # EasyOCRのリーダーを初期化（英語と日本語を指定）
        reader = easyocr.Reader(["en", "ja"], gpu=False)

        # PIL画像をNumPy配列に変換し、OCRを実行
        image_array = np.array(image)
        result = reader.readtext(image_array, detail=0, paragraph=True)
        extracted_text = "\n".join(result)

        st.markdown("### 抽出されたテキスト")
        st.text(extracted_text)

        # テキスト内の数字を抽出して合計を計算
        numbers = re.findall(r"\d+", extracted_text)
        numbers = [int(num) for num in numbers] if numbers else []
        total = sum(numbers)
        st.markdown("### 数字の合計")
        st.write(total)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
else:
    st.info("画像が読み込まれるのを待っています...")
