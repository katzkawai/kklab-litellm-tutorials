# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""20_vllm_embedding.py - vLLMサーバーによる埋め込み生成

事前に埋め込みモデルで vLLM サーバーを起動しておくこと:
  vllm serve intfloat/multilingual-e5-large-instruct --port 8001
"""

from litellm import embedding

VLLM_API_BASE = "http://localhost:8001"

texts = [
    "自然言語処理は人工知能の一分野です。",
    "テキストマイニングで文書を分析します。",
    "株式市場は経済の動向を反映します。",
]

response = embedding(
    model="hosted_vllm/intfloat/multilingual-e5-large-instruct",
    input=texts,
    api_base=VLLM_API_BASE,
)

for i, emb_data in enumerate(response.data):
    vec = emb_data["embedding"]
    print(f"テキスト{i+1}: 「{texts[i][:20]}...」")
    print(f"  次元数: {len(vec)}")
    print(f"  先頭5要素: {vec[:5]}")
    print()
