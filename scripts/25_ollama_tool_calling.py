# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""25_ollama_tool_calling.py - Ollamaでのツール呼び出し

事前に ツール呼び出し対応モデルを取得しておくこと:
  ollama pull llama3.1:8b
"""

import json

from litellm import completion

# ツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "身長と体重からBMIを計算する",
            "parameters": {
                "type": "object",
                "properties": {
                    "height_cm": {
                        "type": "number",
                        "description": "身長（cm）",
                    },
                    "weight_kg": {
                        "type": "number",
                        "description": "体重（kg）",
                    },
                },
                "required": ["height_cm", "weight_kg"],
            },
        },
    }
]


def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return json.dumps({"bmi": round(bmi, 1), "height_cm": height_cm, "weight_kg": weight_kg})


# Step 1: ツール定義付きで呼び出し
messages = [
    {"role": "user", "content": "身長170cm、体重65kgのBMIを計算してください。"}
]

response = completion(
    model="ollama_chat/llama3.1:8b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    api_base="http://localhost:11434",
)

response_message = response.choices[0].message
messages.append(response_message)

# Step 2: ツール呼び出しがあれば実行
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        func_args = json.loads(tool_call.function.arguments)
        print(f"ツール呼び出し: calculate_bmi({func_args})")
        result = calculate_bmi(**func_args)
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": result,
            }
        )

    # Step 3: 最終応答を取得
    final_response = completion(
        model="ollama_chat/llama3.1:8b",
        messages=messages,
        api_base="http://localhost:11434",
    )
    print(f"\n最終回答:\n{final_response.choices[0].message.content}")
else:
    print(f"直接回答:\n{response_message.content}")
