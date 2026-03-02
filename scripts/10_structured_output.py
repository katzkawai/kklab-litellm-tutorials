# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
#     "pydantic",
# ]
# ///
"""10_structured_output.py - Pydanticモデルによる構造化出力"""

import json

from pydantic import BaseModel
from litellm import completion


class BookReview(BaseModel):
    title: str
    author: str
    rating: float
    summary: str
    keywords: list[str]


response = completion(
    model="openai/gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "夏目漱石の『坊っちゃん』の書評を作成してください。"
            ),
        },
    ],
    response_format=BookReview,
)

# Pydanticモデルとしてパース
review = BookReview.model_validate_json(
    response.choices[0].message.content
)

print(f"タイトル: {review.title}")
print(f"著者: {review.author}")
print(f"評価: {review.rating}/5.0")
print(f"要約: {review.summary}")
print(f"キーワード: {', '.join(review.keywords)}")
