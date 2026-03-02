# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""09_json_mode.py - JSONモードによる構造化出力"""

import json

from litellm import completion

# JSONモードを有効にする
response = completion(
    model="openai/gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "あなたはJSON形式で回答するアシスタントです。",
        },
        {
            "role": "user",
            "content": (
                "以下の都市の情報をJSON形式で返してください: "
                "東京、大阪、名古屋。"
                "各都市について name, population（概算）, "
                "famous_for のフィールドを含めてください。"
            ),
        },
    ],
)

# レスポンスをパースして整形出力
raw = response.choices[0].message.content
data = json.loads(raw)
print(json.dumps(data, ensure_ascii=False, indent=2))
