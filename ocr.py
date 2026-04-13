import json
import os
import re
import time

import requests

ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

PROMPT = """\
この画像はテレビ・YouTube動画のテロップ（字幕）部分です。
画像に表示されているテキストをすべて、一文字ずつ丁寧に読み取ってください。

【文字の読み取りで必ず守ること】
- 似ている文字を正確に区別する（例：は／ば／ぱ、う／ぅ、ー／一、り／ソ、ン／ソ、シ／ツ、め／ぬ、る／ろ）
- 濁点（゛）と半濁点（゜）の有無を注意深く確認する
- 漢字は画数・形を正確に読む（例：土／士、己／已、未／末）
- フォントが特殊でも、実際に書かれている文字そのものを読む
- 不明な文字を前後の文脈で補完・推測しない

【誤字チェックのルール】
- 読み取りに少しでも自信がない文字が含まれる場合は has_issue を false にする
- 明らかに間違っているとわかる場合のみ has_issue を true にする
- 固有名詞・商品名・略語は誤字と判定しない

【NG表現チェックのルール】
- 差別用語・放送禁止用語・不適切表現が含まれる場合は has_issue を true、issue_type を "ng" にする
- 誤字・脱字の場合は issue_type を "typo" にする
- 問題がない場合は issue_type を "" にする

出力はJSONのみ（余分なテキスト不要）：
{
  "telop_text": "読み取ったテキスト（テロップがなければ空文字）",
  "has_issue": true または false,
  "issue_type": "typo" または "ng" または "",
  "issue_detail": "問題がある場合のみ説明（なければ空文字）"
}"""


def check_telop(frame_base64, timestamp_seconds):
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        raise Exception('ANTHROPIC_API_KEY が設定されていません。')

    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    payload = {
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 1024,
        'messages': [{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/jpeg',
                        'data': frame_base64,
                    },
                },
                {
                    'type': 'text',
                    'text': PROMPT,
                },
            ],
        }],
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if resp.status_code == 401:
                raise Exception('APIキーが無効です。ANTHROPIC_API_KEY 環境変数を確認してください。')

            if resp.status_code == 429:
                if attempt < 2:
                    time.sleep(10)
                    continue
                raise Exception('Claude API のレート制限に達しました。しばらく待ってから再試行してください。')

            if resp.status_code == 400:
                return None

            if resp.status_code != 200:
                raise Exception(f'Claude API エラー: HTTP {resp.status_code} {resp.text[:200]}')

            data = resp.json()
            text = data['content'][0]['text'].strip()

            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())
            result['timestamp'] = _format_timestamp(timestamp_seconds)
            return result

        except requests.exceptions.ConnectionError as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise Exception(f'Claude API 接続エラー（3回リトライ後）: {str(e)}')
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            raise Exception('Claude API タイムアウト（3回リトライ後）')
        except json.JSONDecodeError:
            return None
        except Exception:
            raise


def check_consistency(telop_texts: list) -> list:
    """
    テロップテキストのリストを受け取り、表記ゆれをまとめて検出する。
    Returns list of {"original": str, "variants": [str], "suggestion": str, "detail": str}
    """
    if not telop_texts:
        return []

    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        return []

    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
    }

    texts_joined = '\n'.join(f'- {t}' for t in telop_texts)
    prompt = f"""\
以下はテレビ・YouTube動画に登場したテロップテキストの一覧です。
表記ゆれ（同じ概念が異なる表記で使われているもの、例：「AI」と「A.I.」、「YouTube」と「ユーチューブ」）を検出してください。

テロップ一覧:
{texts_joined}

【ルール】
- 同じ概念・固有名詞・略語が複数の異なる表記で登場している場合のみ報告する
- 単なる言い換えや文脈が違う表現は表記ゆれとしない
- 表記ゆれがなければ空配列を返す

出力はJSONのみ（余分なテキスト不要）：
[
  {{
    "original": "最初に登場した表記",
    "variants": ["別の表記1", "別の表記2"],
    "suggestion": "推奨する統一表記",
    "detail": "なぜ表記ゆれと判断したかの簡単な説明"
  }}
]"""

    payload = {
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 2048,
        'messages': [{
            'role': 'user',
            'content': prompt,
        }],
    }

    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        text = data['content'][0]['text'].strip()

        # Extract JSON array
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return []

        result = json.loads(json_match.group())
        if isinstance(result, list):
            return result
        return []
    except Exception:
        return []


def _format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'
