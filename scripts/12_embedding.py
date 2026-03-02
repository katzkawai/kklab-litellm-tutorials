# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""12_embedding.py - テキスト埋め込み（Embedding）の生成"""

from litellm import embedding

# 埋め込みの生成
texts = [
    "機械学習は人工知能の一分野です。",
    "深層学習はニューラルネットワークを使用します。",
    "経済学は資源配分を研究する学問です。",
]

response = embedding(
    model="openai/text-embedding-3-small",
    input=texts,
)

# 結果の表示
for i, emb_data in enumerate(response.data):
    vec = emb_data["embedding"]
    print(f"テキスト{i+1}: 「{texts[i][:20]}...」")
    print(f"  次元数: {len(vec)}")
    print(f"  先頭5要素: {vec[:5]}")
    print()

# コサイン類似度の計算
import math


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
