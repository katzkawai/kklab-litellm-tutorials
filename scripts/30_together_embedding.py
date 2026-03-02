# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""30_together_embedding.py - Together AIによる埋め込み生成

事前に環境変数を設定しておくこと:
  export TOGETHERAI_API_KEY="..."
"""

import math

from litellm import embedding

texts = [
    "国内総生産は経済成長の指標である。",
    "GDPは一国の経済規模を測る尺度です。",
    "ニューラルネットワークは深層学習の基盤技術である。",
]

response = embedding(
    model="together_ai/intfloat/multilingual-e5-large-instruct",
    input=texts,
)

# 結果の表示
for i, emb_data in enumerate(response.data):
    vec = emb_data["embedding"]
    print(f"テキスト{i+1}: 「{texts[i][:25]}...」")
    print(f"  次元数: {len(vec)}")
    print()


# コサイン類似度
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


vecs = [d["embedding"] for d in response.data]
print("=== コサイン類似度 ===")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(vecs[i], vecs[j])
        print(f"  テキスト{i+1} vs テキスト{j+1}: {sim:.4f}")
