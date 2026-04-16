import cv2
import numpy as np
import base64

# テロップ領域の差分がこの値を超えたらテロップ変化と判定
TELOP_DIFF_THRESHOLD = 5
# テロップ領域: フレーム下部何%を使うか（上から40%の位置から下を対象）
TELOP_CROP_TOP_RATIO = 0.40
# 差分に関係なく強制的にフレームを含める間隔（秒）
FORCED_INTERVAL_SEC = 5


def extract_key_frames(video_path, progress_callback=None):
    """
    動画から1秒ごとにフレームを抽出し、テロップ領域の差分が大きいフレームだけを返す。

    Returns:
        list of (timestamp_seconds: float, base64_image: str)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(
            '動画ファイルを開けませんでした。ファイルが破損していないか確認してください。'
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise Exception('動画のフレームレートを取得できませんでした。')

    duration = total_frames / fps
    key_frames = []
    prev_telop_gray = None
    second = 0.0

    while True:
        frame_pos = int(second * fps)
        if frame_pos >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

        if progress_callback and duration > 0:
            pct = int((second / duration) * 45)
            progress_callback(pct, f'フレームを抽出中... {int(second)}秒 / {int(duration)}秒')

        # テロップ領域（下部）だけで差分を比較する
        h, w = frame.shape[:2]
        crop_top = int(h * TELOP_CROP_TOP_RATIO)
        telop_area = frame[crop_top:h, 0:w]
        telop_gray = cv2.cvtColor(telop_area, cv2.COLOR_BGR2GRAY)

        include = False
        if prev_telop_gray is None:
            include = True  # 最初のフレームは必ず含める
        else:
            diff = cv2.absdiff(telop_gray, prev_telop_gray)
            if np.mean(diff) > TELOP_DIFF_THRESHOLD:
                include = True
            # 一定間隔で必ず含める（静止テロップを取りこぼさないため）
            elif second % FORCED_INTERVAL_SEC == 0:
                include = True

        if include:
            key_frames.append((second, _frame_to_base64(frame)))

        prev_telop_gray = telop_gray
        second += 1.0

    cap.release()

    if progress_callback:
        progress_callback(
            50,
            f'フレーム抽出完了: {len(key_frames)} フレームをテロップ解析します'
        )

    return key_frames


def _frame_to_base64(frame):
    """
    テロップ認識用にフレームを処理してbase64に変換する。
    - 下部40%を切り出してテロップ領域に集中
    - 切り出し部分を拡大してテキストを読みやすくする
    - JPEG品質92で高精度に保存
    """
    h, w = frame.shape[:2]

    # テロップは主に下部に表示されるため下部40%を切り出し
    crop_top = int(h * 0.55)
    telop_region = frame[crop_top:h, 0:w]

    # 切り出し部分を幅1280pxに拡大（小さすぎる場合のみ拡大）
    th, tw = telop_region.shape[:2]
    target_width = 1280
    if tw < target_width:
        scale = target_width / tw
        telop_region = cv2.resize(
            telop_region,
            (target_width, int(th * scale)),
            interpolation=cv2.INTER_CUBIC
        )
    elif tw > target_width:
        scale = target_width / tw
        telop_region = cv2.resize(
            telop_region,
            (target_width, int(th * scale)),
            interpolation=cv2.INTER_AREA
        )

    # CLAHE contrast enhancement on L channel (LAB color space)
    lab = cv2.cvtColor(telop_region, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    telop_region = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    telop_region = cv2.filter2D(telop_region, -1, kernel)

    _, buffer = cv2.imencode('.jpg', telop_region, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')
