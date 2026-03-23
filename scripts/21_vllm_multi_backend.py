# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""21_vllm_multi_backend.py - vLLMとクラウドAPIの混合利用

ローカルの vLLM サーバーとクラウド API を同じコードで切り替える例。
Router を使えば、ローカルモデルを優先しつつクラウドにフォールバックできる。

事前に vLLM サーバーを起動しておくこと:
  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
"""

import asyncio
import os
import time

from litellm import Router

# ローカル vLLM + クラウド API の混合構成
model_list = [
    {
        "model_name": "my-llm",
        "litellm_params": {
            "model": "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
            "api_base": "http://localhost:8000",
            "rpm": 100,   # ローカルなので高い RPM を設定
        },
        "model_info": {
            "description": "ローカル vLLM サーバー（優先）",
        },
    },
    {
        "model_name": "my-llm-cloud",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {
            "description": "クラウド API（フォールバック）",
        },
    },
]

router = Router(
    model_list=model_list,
    # 通常はローカル vLLM を使い、失敗時だけクラウドへ切り替える。
    fallbacks=[{"my-llm": ["my-llm-cloud"]}],
    num_retries=2,
)


async def main():
    question = "回帰分析とは何ですか？一文で簡潔に答えてください。"

    print("=== vLLM + クラウド API 混合構成 ===\n")
    start = time.time()

    # Router がローカル vLLM を優先し、失敗時にクラウドへフォールバック
    response = await router.acompletion(
        model="my-llm",
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )

    elapsed = time.time() - start
    print(f"使用モデル: {response.model}")
    print(f"回答: {response.choices[0].message.content}")
    print(f"応答時間: {elapsed:.2f}秒")


asyncio.run(main())
