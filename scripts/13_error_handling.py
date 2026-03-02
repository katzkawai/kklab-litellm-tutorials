# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""13_error_handling.py - 例外処理とエラーハンドリング"""

import litellm
from litellm import completion


def safe_completion(model: str, messages: list, **kwargs) -> str | None:
    """エラーハンドリング付きの補完関数"""
    try:
        response = completion(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content

    except litellm.AuthenticationError as e:
        print(f"認証エラー: APIキーを確認してください - {e}")

    except litellm.RateLimitError as e:
        print(f"レート制限: しばらく待ってから再試行してください - {e}")

    except litellm.ContextWindowExceededError as e:
        print(f"コンテキスト長超過: 入力を短くしてください - {e}")

    except litellm.BadRequestError as e:
        print(f"不正なリクエスト: {e}")

    except litellm.APIConnectionError as e:
        print(f"API接続エラー: ネットワークを確認してください - {e}")

    except litellm.Timeout as e:
        print(f"タイムアウト: {e}")

    except Exception as e:
        print(f"予期しないエラー: {type(e).__name__}: {e}")

    return None


# 正常なリクエスト
print("=== 正常なリクエスト ===")
result = safe_completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "こんにちは"}],
)
if result:
    print(f"応答: {result}\n")

# タイムアウトの例（非常に短いタイムアウト）
print("=== タイムアウトの例 ===")
result = safe_completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "長い文章を書いてください。"}],
    timeout=0.001,
)
