# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""19_vllm_streaming.py - vLLMサーバーからのストリーミング応答

事前に vLLM サーバーを起動しておくこと:
  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
"""

from litellm import completion

VLLM_API_BASE = "http://localhost:8000"

print("=== vLLM ストリーミング ===")
response = completion(
    model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "日本の経済史を3つの時代に分けて簡潔に説明してください。"},
    ],
    api_base=VLLM_API_BASE,
    stream=True,
    max_tokens=500,
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()
