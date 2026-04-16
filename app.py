import os
import uuid
import csv
import io
import threading
from datetime import datetime

from flask import Flask, request, jsonify, render_template, Response
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

COOKIES_PATH = os.path.join(os.path.dirname(__file__), 'yt_cookies.txt')


def _ensure_cookies_file():
    """環境変数 YT_COOKIES_B64 があればファイルに書き出す"""
    b64 = os.environ.get('YT_COOKIES_B64', '').strip()
    if b64 and not os.path.exists(COOKIES_PATH):
        import base64
        try:
            with open(COOKIES_PATH, 'wb') as f:
                f.write(base64.b64decode(b64))
        except Exception:
            pass

# タスクの状態をメモリ上で管理
tasks = {}
# チェック履歴（最大50件）
history = []


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
        'filename': file.filename,
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
    filename   = request.form.get('filename', 'unknown.mp4')

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
        'filename': filename,
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
        from frame_extractor import extract_key_frames
        from ocr import check_telop, check_consistency

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
                'consistency_issues': [],
            })
            _add_to_history(task_id)
            return

        results = []
        total = len(frames)
        prev_text = None

        for i, (timestamp, frame_b64) in enumerate(frames):
            tasks[task_id]['progress'] = 50 + int((i + 1) / total * 45)
            tasks[task_id]['message'] = f'テロップを解析中... ({i + 1} / {total} フレーム)'

            result = check_telop(frame_b64, timestamp)

            if result and result.get('telop_text', '').strip():
                if result['telop_text'] != prev_text:
                    results.append(result)
                    prev_text = result['telop_text']

        # 表記ゆれチェック
        tasks[task_id]['progress'] = 96
        tasks[task_id]['message'] = '表記ゆれをチェック中...'
        telop_texts = [r['telop_text'] for r in results if r.get('telop_text', '').strip()]
        consistency_result = check_consistency(telop_texts)

        tasks[task_id].update({
            'status': 'done',
            'progress': 100,
            'message': '処理が完了しました',
            'results': results,
            'consistency_issues': consistency_result,
        })
        _add_to_history(task_id)

    except Exception as e:
        tasks[task_id].update({
            'status': 'error',
            'message': f'エラーが発生しました: {str(e)}',
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/upload_cookies', methods=['POST'])
def upload_cookies():
    import base64
    try:
        if 'cookies' not in request.files:
            return jsonify({'error': 'ファイルが見つかりません'}), 400
        file = request.files['cookies']
        if not file.filename.lower().endswith('.txt'):
            return jsonify({'error': '.txtファイルを選択してください'}), 400
        content = file.read()
        # ファイルに保存
        with open(COOKIES_PATH, 'wb') as f:
            f.write(content)
        # 環境変数用のBase64も返す（Railwayに手動設定してもらうため）
        b64 = base64.b64encode(content).decode('utf-8')
        return jsonify({'ok': True, 'message': 'Cookieを保存しました', 'b64': b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/cookie_status')
def cookie_status():
    b64 = os.environ.get('YT_COOKIES_B64', '').strip()
    _ensure_cookies_file()
    return jsonify({
        'has_cookies': os.path.exists(COOKIES_PATH),
        'has_b64_env': bool(b64),
        'b64_length': len(b64),
    })


@app.route('/start_youtube', methods=['POST'])
def start_youtube():
    data = request.get_json(force=True)
    url = (data or {}).get('url', '').strip()
    if not url:
        return jsonify({'error': 'URLを指定してください。'}), 400

    if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
        return jsonify({'error': 'ANTHROPIC_API_KEY が設定されていません。'}), 500

    task_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')

    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'ダウンロードを開始しています...',
        'results': [],
        'filename': url,
    }

    thread = threading.Thread(
        target=process_youtube,
        args=(task_id, url, filepath),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/start_gdrive', methods=['POST'])
def start_gdrive():
    data = request.get_json(force=True)
    url = (data or {}).get('url', '').strip()
    if not url:
        return jsonify({'error': 'URLを指定してください。'}), 400

    if 'drive.google.com' not in url:
        return jsonify({'error': '有効なGoogle Drive URLを指定してください。'}), 400

    if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
        return jsonify({'error': 'ANTHROPIC_API_KEY が設定されていません。'}), 500

    task_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f'{task_id}.mp4')

    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'ダウンロードを開始しています...',
        'results': [],
        'filename': url,
    }

    thread = threading.Thread(
        target=process_gdrive,
        args=(task_id, url, filepath),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def process_gdrive(task_id, url, filepath):
    try:
        import gdown
        tasks[task_id]['message'] = 'Google Driveから動画をダウンロード中...'
        tasks[task_id]['progress'] = 10

        # Google Drive URLからファイルIDを抽出して直接ダウンロードURLに変換
        import re as _re
        match = _re.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if match:
            file_id = match.group(1)
            download_url = f'https://drive.google.com/uc?id={file_id}'
        else:
            download_url = url
        output = gdown.download(download_url, filepath, quiet=True)
        if not output or not os.path.exists(filepath):
            raise Exception('ダウンロードに失敗しました。共有設定が「リンクを知っている全員」になっているか確認してください。')

        tasks[task_id]['message'] = 'ダウンロード完了。処理を開始します...'
        tasks[task_id]['progress'] = 30

        # Extract filename from URL or use default
        tasks[task_id]['filename'] = 'Google Drive動画'

        process_video(task_id, filepath)
    except Exception as e:
        err = str(e)
        if 'permission' in err.lower() or 'access' in err.lower() or '403' in err:
            msg = '⚠️ アクセスできませんでした。\nGoogleドライブの共有設定を「リンクを知っている全員が閲覧可能」に変更してください。'
        else:
            msg = f'Google Driveダウンロードエラー: {err}'
        tasks[task_id].update({'status': 'error', 'message': msg})
        if os.path.exists(filepath):
            os.remove(filepath)


def process_youtube(task_id, url, filepath):
    try:
        import yt_dlp

        def progress_hook(d):
            if d['status'] == 'downloading':
                pct_raw = d.get('_percent_str', '').strip()
                # Parse percentage number from string like "45.2%"
                try:
                    pct_num = float(pct_raw.replace('%', '').strip())
                    tasks[task_id]['progress'] = int(pct_num * 0.3)  # scale to 0-30
                except Exception:
                    pass
                tasks[task_id]['message'] = f'ダウンロード中... {pct_raw}'

        # 一時ファイルはtask_idのみ（拡張子なし）でyt-dlpに任せる
        tmp_base = os.path.join(UPLOAD_FOLDER, task_id)

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            'outtmpl': tmp_base + '.%(ext)s',
            'quiet': True,
            'progress_hooks': [progress_hook],
            'extractor_args': {'youtube': {'player_client': ['ios', 'web']}},
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.29.1 (iPhone16,2; U; CPU iOS 17_5_1 like Mac OS X;)',
            },
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        _ensure_cookies_file()
        if os.path.exists(COOKIES_PATH):
            ydl_opts['cookiefile'] = COOKIES_PATH

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', url) if info else url
            tasks[task_id]['filename'] = title

        # ダウンロードされたファイルを探してfilepathに統一
        import glob
        candidates = glob.glob(tmp_base + '.*')
        if not candidates:
            raise Exception('動画ファイルのダウンロードに失敗しました。')
        downloaded = candidates[0]
        if downloaded != filepath:
            os.rename(downloaded, filepath)

        # After download, process the video
        process_video(task_id, filepath)

    except Exception as e:
        err = str(e)
        if 'Sign in' in err or 'bot' in err or 'cookies' in err:
            msg = '⚠️ この動画はダウンロードできませんでした。\n限定公開・非公開動画はサーバーからアクセスできません。\nMP4ファイルをダウンロードして「ファイルアップロード」タブから試してください。'
        elif 'format' in err.lower():
            msg = '動画フォーマットの取得に失敗しました。別の動画で試してみてください。'
        else:
            msg = f'YouTubeダウンロードエラー: {err}'
        tasks[task_id].update({
            'status': 'error',
            'message': msg,
        })


def _add_to_history(task_id):
    """完了タスクを履歴に追加する"""
    task = tasks.get(task_id)
    if not task:
        return
    results = task.get('results', [])
    history.insert(0, {
        'task_id': task_id,
        'filename': task.get('filename', '不明'),
        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total': len(results),
        'issues': sum(1 for r in results if r.get('has_issue')),
    })
    # 最大50件まで保持
    if len(history) > 50:
        history.pop()


@app.route('/history')
def get_history():
    return jsonify(history)


@app.route('/test-api')
def test_api():
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
