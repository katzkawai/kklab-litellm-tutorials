# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""23_ollama_streaming.py - Ollamaからのストリーミング応答

事前に Ollama でモデルを取得しておくこと:
  ollama pull llama3.2:3b
"""

from litellm import completion

print("=== Ollama ストリーミング ===")
response = completion(
    model="ollama_chat/llama3.2:3b",
    messages=[
        {"role": "user", "content": "日本の主要な統計指標を5つ挙げて、それぞれ一文で説明してください。"},
    ],
    api_base="http://localhost:11434",
    stream=True,
    max_tokens=500,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()
