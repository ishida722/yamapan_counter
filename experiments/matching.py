import cv2
import numpy as np
import glob
import os
import argparse

def match_template_orb(main_img, template, ratio_threshold=0.75):
    """
    ORB特徴量を利用してテンプレート画像を入力画像内でマッチングし、
    ホモグラフィからテンプレート領域の位置を推定します。
    """
    # グレースケール変換
    main_gray = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY) if len(main_img.shape) == 3 else main_img
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

    # ORB特徴量検出器生成
    orb = cv2.ORB_create(5000)
    kp_template, des_template = orb.detectAndCompute(template_gray, None)
    kp_main, des_main = orb.detectAndCompute(main_gray, None)

    # BFMatcherの生成 (ORBはバイナリ特徴量)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des_template, des_main, k=2)

    # Loweの比率テスト
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # 十分なマッチング数がなければ終了
    if len(good_matches) < 4:
        return None

    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_main[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィ推定（RANSACを使用）
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None

    h, w = template_gray.shape
    pts = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    return dst, len(good_matches)

def main(args):
    main_img = cv2.imread(args.input)
    if main_img is None:
        print("入力画像が読み込めませんでした:", args.input)
        return

    template_paths = glob.glob(os.path.join(args.templates, "*.png"))
    if not template_paths:
        print("テンプレート画像が見つかりませんでした:", args.templates)
        return

    output_img = main_img.copy()
    for tpl_path in template_paths:
        template = cv2.imread(tpl_path)
        if template is None:
            print("テンプレート画像の読み込み失敗:", tpl_path)
            continue

        result = match_template_orb(main_img, template, ratio_threshold=args.ratio)
        if result is not None:
            dst, good_count = result
            dst = dst.astype(int)
            cv2.polylines(output_img, [dst], True, (0, 255, 0), 3, cv2.LINE_AA)
            print(f"[INFO] {tpl_path} : マッチ成功（良い特徴点数 = {good_count}）")
        else:
            print(f"[INFO] {tpl_path} : マッチする特徴点が不足しています。")

    cv2.imshow("Detection Result", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORB特徴量を用いたテンプレートマッチング")
    parser.add_argument("--input", required=True, help="入力画像のパス")
    parser.add_argument("--templates", required=True, help="テンプレート画像が格納されたディレクトリ")
    parser.add_argument("--ratio", type=float, default=0.75, help="Loweの比率テスト用の閾値")
    args = parser.parse_args()
    main(args)