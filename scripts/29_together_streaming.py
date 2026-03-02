# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""29_together_streaming.py - Together AIからのストリーミング応答

事前に環境変数を設定しておくこと:
  export TOGETHERAI_API_KEY="..."
"""

from litellm import completion

print("=== Together AI ストリーミング (DeepSeek) ===")
response = completion(
    model="together_ai/deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "user", "content": "パネルデータ分析の利点を3つ挙げて説明してください。"},
    ],
    stream=True,
    max_tokens=500,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()
