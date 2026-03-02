# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""15_router.py - Routerによるロードバランシング"""

import os
import asyncio

from litellm import Router

# 複数デプロイメントの設定
model_list = [
    {
        "model_name": "my-gpt",  # エイリアス名
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
    {
        "model_name": "my-gpt",  # 同じエイリアスで別モデルも設定可能
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    },
]

# Routerの初期化
router = Router(
    model_list=model_list,
    num_retries=3,            # リトライ回数
    fallbacks=[               # フォールバック設定
        {"my-gpt": ["my-gpt"]}
    ],
    routing_strategy="simple-shuffle",  # ロードバランシング戦略
)


async def main():
    # 複数リクエストを並列実行
    tasks = []
    for i in range(3):
        tasks.append(
            router.acompletion(
                model="my-gpt",
                messages=[
                    {
                        "role": "user",
                        "content": f"数字の{i+1}に関する豆知識を一つ。",
                    }
                ],
                max_tokens=100,
            )
        )

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            print(f"リクエスト{i+1}: エラー - {resp}")
        else:
            print(f"リクエスト{i+1}: {resp.choices[0].message.content}\n")


asyncio.run(main())
