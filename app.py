import os
import uuid
import csv
import io
import threading

from flask import Flask, request, jsonify, render_template, Response
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# タスクの状態をメモリ上で管理
tasks = {}


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return jsonify({
        'error': 'ファイルサイズが1GBを超えています。1GB以下のファイルをアップロードしてください。'
    }), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'ファイルが見つかりません。動画ファイルを選択してください。'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません。'}), 400

    if not file.filename.lower().endswith('.mp4'):
        return jsonify({'error': 'MP4ファイルのみ対応しています。別の形式の場合はMP4に変換してください。'}), 400

    if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
        return jsonify({'error': 'ANTHROPIC_API_KEY が設定されていません。.env ファイルまたは環境変数を確認してください。'}), 500

    task_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')
    file.save(filepath)

    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': '処理を開始しています...',
        'results': [],
    }

    thread = threading.Thread(
        target=process_video,
        args=(task_id, filepath),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    import shutil
    task_id    = request.form.get('task_id')
    chunk_idx  = request.form.get('chunk_index')
    total      = request.form.get('total_chunks')
    chunk_file = request.files.get('chunk')

    if not task_id or chunk_idx is None or total is None or chunk_file is None:
        return jsonify({'error': 'パラメータが不正です'}), 400

    chunk_idx = int(chunk_idx)
    total     = int(total)

    chunk_dir = os.path.join(UPLOAD_FOLDER, f'{task_id}_chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file.save(os.path.join(chunk_dir, f'{chunk_idx:06d}'))

    if chunk_idx < total - 1:
        return jsonify({'received': chunk_idx})

    # 全チャンク受信完了 → 結合
    filepath = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')
    with open(filepath, 'wb') as out:
        for i in range(total):
            p = os.path.join(chunk_dir, f'{i:06d}')
            with open(p, 'rb') as f:
                out.write(f.read())
    shutil.rmtree(chunk_dir, ignore_errors=True)

    if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
        os.remove(filepath)
        return jsonify({'error': 'ANTHROPIC_API_KEY が設定されていません。'}), 500

    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': '処理を開始しています...',
        'results': [],
    }

    thread = threading.Thread(
        target=process_video,
        args=(task_id, filepath),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'processing'})


def process_video(task_id, filepath):
    try:
        from extractor import extract_key_frames
        from ocr import check_telop

        def progress_cb(p, msg):
            tasks[task_id]['progress'] = p
            tasks[task_id]['message'] = msg

        tasks[task_id]['message'] = 'フレームを抽出しています...'
        frames = extract_key_frames(filepath, progress_callback=progress_cb)

        if not frames:
            tasks[task_id].update({
                'status': 'done',
                'progress': 100,
                'message': '処理が完了しました（テロップが検出されませんでした）',
                'results': [],
            })
            return

        results = []
        total = len(frames)
        prev_text = None

        for i, (timestamp, frame_b64) in enumerate(frames):
            tasks[task_id]['progress'] = 50 + int((i + 1) / total * 50)
            tasks[task_id]['message'] = f'テロップを解析中... ({i + 1} / {total} フレーム)'

            result = check_telop(frame_b64, timestamp)

            if result and result.get('telop_text', '').strip():
                # 連続する同一テロップは重複除去
                if result['telop_text'] != prev_text:
                    results.append(result)
                    prev_text = result['telop_text']

        tasks[task_id].update({
            'status': 'done',
            'progress': 100,
            'message': '処理が完了しました',
            'results': results,
        })

    except Exception as e:
        tasks[task_id].update({
            'status': 'error',
            'message': f'エラーが発生しました: {str(e)}',
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/test-api')
def test_api():
    """Anthropic API への接続テスト用エンドポイント"""
    import requests as _requests

    results = {}
    key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    results['api_key_set'] = bool(key)
    results['api_key_prefix'] = key[:12] if key else 'none'

    if key:
        try:
            resp = _requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json',
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'hi'}],
                },
                timeout=20,
            )
            if resp.status_code == 200:
                results['api'] = f'ok: {resp.json()["content"][0]["text"]}'
            else:
                results['api'] = f'HTTP {resp.status_code}: {resp.text[:200]}'
        except Exception as e:
            results['api'] = f'NG {type(e).__name__}: {str(e)}'

    return jsonify(results)


@app.route('/status/<task_id>')
def status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'タスクが見つかりません'}), 404
    return jsonify(tasks[task_id])


@app.route('/download/<task_id>')
def download_csv(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'タスクが見つかりません'}), 404

    task = tasks[task_id]
    if task['status'] != 'done':
        return jsonify({'error': 'まだ処理が完了していません'}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['タイムスタンプ', 'テロップ内容', '問題あり', '問題の詳細'])

    for r in task['results']:
        writer.writerow([
            r.get('timestamp', ''),
            r.get('telop_text', ''),
            '有' if r.get('has_issue') else '無',
            r.get('issue_detail', ''),
        ])

    output.seek(0)

    # UTF-8 BOM付き（Excel で文字化けしない）
    return Response(
        '\ufeff' + output.getvalue(),
        mimetype='text/csv; charset=utf-8-sig',
        headers={
            'Content-Disposition': 'attachment; filename=telop_check_results.csv'
        }
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', debug=False, port=port)
