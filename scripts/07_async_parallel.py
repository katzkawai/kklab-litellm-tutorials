# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""07_async_parallel.py - 非同期による複数モデルの並列呼び出し"""

import asyncio
import time

from litellm import acompletion


async def call_model(model: str, question: str) -> dict:
    """単一モデルへの非同期呼び出し"""
    start = time.time()
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    elapsed = time.time() - start
    return {
        "model": model,
        "response": response.choices[0].message.content,
        "time": elapsed,
        "tokens": response.usage.total_tokens,
    }


async def main():
    models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-haiku-4-5-20251001",
        "gemini/gemini-2.0-flash",
    ]
    question = "データサイエンスで最も重要なスキルは何ですか？一文で答えてください。"

    print("=== 複数モデルへの並列リクエスト ===\n")
    start = time.time()

    # 全モデルに並列でリクエスト
    results = await asyncio.gather(
        *[call_model(m, question) for m in models],
        return_exceptions=True,
    )

    total_time = time.time() - start

    for result in results:
        if isinstance(result, Exception):
            print(f"エラー: {result}\n")
        else:
            print(f"[{result['model']}] ({result['time']:.2f}秒)")
            print(f"  回答: {result['response']}")
            print(f"  トークン数: {result['tokens']}\n")

    print(f"全体の実行時間: {total_time:.2f}秒")


asyncio.run(main())
