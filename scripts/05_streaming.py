# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""05_streaming.py - ストリーミング応答"""

from litellm import completion

print("=== 同期ストリーミング ===")
response = completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "日本の四季について短く説明してください。"}],
    stream=True,
)

# チャンクごとにリアルタイムで出力
for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print("\n")
