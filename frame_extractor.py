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
    フレーム全体をClaudeに送信する。
    テロップがどの位置にあっても検出できるよう切り出しは行わない。
    幅1280pxにリサイズしてJPEG品質92で保存。
    """
    h, w = frame.shape[:2]

    # 幅1280pxに統一（縦横比維持）
    target_width = 1280
    if w != target_width:
        scale = target_width / w
        frame = cv2.resize(
            frame,
            (target_width, int(h * scale)),
            interpolation=cv2.INTER_AREA if w > target_width else cv2.INTER_CUBIC
        )

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')
