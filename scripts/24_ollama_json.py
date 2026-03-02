# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""24_ollama_json.py - OllamaでのJSONモード

事前に Ollama でモデルを取得しておくこと:
  ollama pull llama3.2:3b
"""

import json

from litellm import completion

response = completion(
    model="ollama_chat/llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": "あなたはJSON形式で回答するアシスタントです。",
        },
        {
            "role": "user",
            "content": (
                "以下のプログラミング言語の情報をJSON配列で返してください: "
                "Python, R, Julia。"
                "各言語について name, paradigm, "
                "main_use のフィールドを含めてください。"
            ),
        },
    ],
    api_base="http://localhost:11434",
    format="json",
    max_tokens=500,
)

raw = response.choices[0].message.content
data = json.loads(raw)
print(json.dumps(data, ensure_ascii=False, indent=2))
