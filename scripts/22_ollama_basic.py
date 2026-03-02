# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""22_ollama_basic.py - Ollamaによるローカルモデルの呼び出し

事前に Ollama をインストールし、モデルを取得しておくこと:
  ollama pull llama3.2:3b
"""

from litellm import completion

# Ollama はデフォルトで http://localhost:11434 で起動する
response = completion(
    model="ollama_chat/llama3.2:3b",
    messages=[
        {"role": "system", "content": "あなたは親切なアシスタントです。日本語で回答してください。"},
        {"role": "user", "content": "Pythonでフィボナッチ数列を生成する方法を教えてください。"},
    ],
    api_base="http://localhost:11434",
    temperature=0.7,
    max_tokens=500,
)

print("=== Ollama レスポンス ===")
print(response.choices[0].message.content)
print(f"\nモデル: {response.model}")
print(f"トークン: {response.usage.total_tokens}")
