# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""17_cost_tracking.py - コスト追跡"""

import litellm
from litellm import completion, completion_cost

# コスト追跡の例
models_and_prompts = [
    ("openai/gpt-4o-mini", "こんにちは"),
    ("openai/gpt-4o-mini", "Pythonの利点を5つ挙げてください。"),
    ("anthropic/claude-haiku-4-5-20251001", "機械学習とは何ですか？"),
]

total_cost = 0.0

for model, prompt in models_and_prompts:
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        # コスト計算
        cost = completion_cost(completion_response=response)
        total_cost += cost

        print(f"モデル: {model}")
        print(f"  入力トークン: {response.usage.prompt_tokens}")
        print(f"  出力トークン: {response.usage.completion_tokens}")
        print(f"  コスト: ${cost:.6f}")
        print()

    except Exception as e:
        print(f"モデル: {model} - エラー: {e}\n")

print(f"=== 合計コスト: ${total_cost:.6f} ===")
