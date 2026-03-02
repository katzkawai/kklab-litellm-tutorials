# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""27_together_basic.py - Together AIによるオープンソースモデルの呼び出し

事前に環境変数を設定しておくこと:
  export TOGETHERAI_API_KEY="..."
"""

from litellm import completion

# Together AI 上の Llama モデルを呼び出す
response = completion(
    model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[
        {"role": "system", "content": "あなたは親切なアシスタントです。日本語で回答してください。"},
        {"role": "user", "content": "ベイズ統計学の基本的な考え方を説明してください。"},
    ],
    temperature=0.7,
    max_tokens=500,
)

print("=== Together AI (Llama 3.1) レスポンス ===")
print(response.choices[0].message.content)
print(f"\nモデル: {response.model}")
print(f"入力トークン: {response.usage.prompt_tokens}")
print(f"出力トークン: {response.usage.completion_tokens}")
