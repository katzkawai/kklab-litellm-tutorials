# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""14_fallback.py - フォールバック（代替モデルへの切り替え）"""

from litellm import completion

# フォールバックリスト: メインモデルが失敗したら順に試行
fallback_models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5-20251001",
    "gemini/gemini-2.0-flash",
]


def completion_with_fallback(messages: list, **kwargs) -> str:
    """フォールバック付き補完関数"""
    errors = []
    for model in fallback_models:
        try:
            print(f"  試行中: {model}")
            response = completion(
                model=model, messages=messages, **kwargs
            )
            print(f"  成功: {model}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"  失敗: {model} - {e}")
            errors.append((model, e))

    raise RuntimeError(
        f"全モデルが失敗しました: {errors}"
    )


# 実行例
messages = [
    {"role": "user", "content": "量子コンピュータとは何ですか？一文で。"}
]
print("=== フォールバック実行 ===")
result = completion_with_fallback(messages, max_tokens=200)
print(f"\n回答: {result}")
