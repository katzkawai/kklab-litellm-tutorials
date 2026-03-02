# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""03_system_prompt.py - システムプロンプトの活用"""

from litellm import completion

# システムプロンプトでモデルの振る舞いを制御する
messages = [
    {
        "role": "system",
        "content": (
            "あなたは経済学の教授です。"
            "専門用語はなるべく平易な言葉で説明し、"
            "具体例を交えて回答してください。"
        ),
    },
    {
        "role": "user",
        "content": "インフレーションとは何ですか？",
    },
]

response = completion(
    model="openai/gpt-4o-mini",
    messages=messages,
    temperature=0.7,
    max_tokens=500,
)

print(response.choices[0].message.content)
