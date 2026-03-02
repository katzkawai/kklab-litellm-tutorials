# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""04_multi_turn.py - マルチターン会話"""

from litellm import completion

# 会話履歴を保持するリスト
messages = [
    {
        "role": "system",
        "content": "あなたは親切なプログラミング講師です。",
    },
]


def chat(user_message: str) -> str:
    """ユーザーメッセージを送信し、応答を返す"""
    messages.append({"role": "user", "content": user_message})

    response = completion(
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message


# マルチターン会話の実行
turns = [
    "Pythonでリストの基本操作を教えてください。",
    "リスト内包表記についても教えてください。",
    "では、それを使った実践的な例を示してください。",
]

for i, user_msg in enumerate(turns, 1):
    print(f"--- ターン {i} ---")
    print(f"ユーザー: {user_msg}")
    reply = chat(user_msg)
    print(f"アシスタント: {reply}\n")
