# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""11_tool_calling.py - ツール呼び出し（Function Calling）"""

import json

from litellm import completion

# ツール（関数）の定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された都市の現在の天気を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "都市名（例: 東京）",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度の単位",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


# ダミーの天気関数
def get_weather(city: str, unit: str = "celsius") -> str:
    """実際のAPIの代わりにダミーデータを返す"""
    weather_data = {
        "東京": {"temp": 22, "condition": "晴れ"},
        "大阪": {"temp": 24, "condition": "曇り"},
        "名古屋": {"temp": 23, "condition": "晴れ時々曇り"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "不明"})
    return json.dumps(
        {"city": city, "temperature": data["temp"],
         "unit": unit, "condition": data["condition"]},
        ensure_ascii=False,
    )


# Step 1: ツール定義付きでモデルを呼び出す
messages = [
    {"role": "user", "content": "東京と名古屋の天気を教えてください。"}
]

response = completion(
    model="openai/gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

response_message = response.choices[0].message
messages.append(response_message)

# Step 2: ツール呼び出しを実行
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        print(f"ツール呼び出し: {func_name}({func_args})")

        # 関数を実行
        result = get_weather(**func_args)

        # 結果をメッセージに追加
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": result,
            }
        )

    # Step 3: ツール結果を含めて再度モデルを呼び出す
    final_response = completion(
        model="openai/gpt-4o-mini",
        messages=messages,
    )

    print(f"\n最終回答:\n{final_response.choices[0].message.content}")
