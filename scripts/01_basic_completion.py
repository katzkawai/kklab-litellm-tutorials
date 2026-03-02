# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""01_basic_completion.py - LiteLLMの基本的な補完呼び出し"""

import os
from litellm import completion

# 環境変数からAPIキーを読み込む（事前に設定しておくこと）
# export OPENAI_API_KEY="sk-..."

response = completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Pythonの特徴を3つ挙げてください。"}],
)

print("=== レスポンス内容 ===")
print(response.choices[0].message.content)
print("\n=== トークン使用量 ===")
print(f"入力トークン: {response.usage.prompt_tokens}")
print(f"出力トークン: {response.usage.completion_tokens}")
print(f"合計トークン: {response.usage.total_tokens}")
print(f"\n=== モデル情報 ===")
print(f"モデル: {response.model}")
