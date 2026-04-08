import json
import os
import re
import time

import anthropic

PROMPT = """\
この動画フレームに表示されているテロップ（画面に重ねられたテキスト）を
すべて読み取ってください。その後、日本語として誤字・脱字・
表記ゆれがないかチェックし、問題があれば指摘してください。
出力はJSON形式で：
{
  "telop_text": "読み取ったテキスト",
  "has_issue": true/false,
  "issue_detail": "問題の説明（問題なければ空文字）"
}

テロップが存在しない場合は telop_text を空文字にしてください。
JSONのみを出力してください。余分なテキストは不要です。"""


def check_telop(frame_base64, timestamp_seconds):
    """
    フレーム画像を Claude Vision API に送り、テロップの読み取りと誤字チェックを行う。

    Returns:
        dict with keys: timestamp, telop_text, has_issue, issue_detail
        テロップが無い場合は None
    """
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    for attempt in range(3):
        try:
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=1024,
                messages=[{
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
                }]
            )

            text = response.content[0].text.strip()

            # レスポンスから JSON を抽出
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            data['timestamp'] = _format_timestamp(timestamp_seconds)
            return data

        except json.JSONDecodeError:
            return None
        except anthropic.AuthenticationError:
            raise Exception(
                'APIキーが無効です。ANTHROPIC_API_KEY 環境変数を確認してください。'
            )
        except anthropic.RateLimitError:
            # レート制限は少し待ってリトライ
            if attempt < 2:
                time.sleep(10)
                continue
            raise Exception(
                'Claude API のレート制限に達しました。しばらく待ってから再試行してください。'
            )
        except anthropic.BadRequestError:
            # 画像解析できないフレームはスキップ
            return None
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            # 接続エラーはリトライ
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise Exception(f'Claude API 接続エラー（3回リトライ後）: {str(e)}')
        except Exception as e:
            raise Exception(f'Claude API エラー: {str(e)}')


def _format_timestamp(seconds):
    """秒数を MM:SS または HH:MM:SS 形式に変換する。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'
