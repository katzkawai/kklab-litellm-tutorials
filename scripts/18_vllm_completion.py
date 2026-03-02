# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""18_vllm_completion.py - vLLMサーバー経由の補完呼び出し

事前に vLLM サーバーを起動しておくこと:
  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
"""

from litellm import completion

# vLLM サーバーのベースURL（デフォルト: http://localhost:8000）
VLLM_API_BASE = "http://localhost:8000"

response = completion(
    model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "Pythonのリスト内包表記について簡潔に説明してください。"},
    ],
    api_base=VLLM_API_BASE,
    temperature=0.7,
    max_tokens=300,
)

print("=== vLLM レスポンス ===")
print(response.choices[0].message.content)
print(f"\nモデル: {response.model}")
print(f"入力トークン: {response.usage.prompt_tokens}")
print(f"出力トークン: {response.usage.completion_tokens}")
