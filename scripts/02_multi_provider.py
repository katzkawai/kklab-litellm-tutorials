# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""02_multi_provider.py - 複数プロバイダーの統一的な呼び出し"""

import os
from litellm import completion

# 使用するモデルのリスト（プロバイダー/モデル名の形式）
models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5-20251001",
    "gemini/gemini-2.0-flash",
]

question = "機械学習とは何ですか？一文で説明してください。"

for model_name in models:
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=200,
        )
        print(f"[{model_name}]")
        print(f"  回答: {response.choices[0].message.content}")
        print(f"  トークン: {response.usage.total_tokens}")
        print()
    except Exception as e:
        print(f"[{model_name}] エラー: {e}\n")
