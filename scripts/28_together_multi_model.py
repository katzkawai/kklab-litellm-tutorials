# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""28_together_multi_model.py - Together AI上の複数オープンソースモデル比較

事前に環境変数を設定しておくこと:
  export TOGETHERAI_API_KEY="..."
"""

import asyncio
import time

from litellm import acompletion


async def call_model(model: str, question: str) -> dict:
    """モデルを呼び出して結果を返す"""
    start = time.time()
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=200,
        )
        elapsed = time.time() - start
        return {
            "model": model,
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "time": elapsed,
        }
    except Exception as e:
        return {"model": model, "response": None, "tokens": 0,
                "time": time.time() - start, "error": str(e)}


async def main():
    # Together AI 上の異なるオープンソースモデルを比較
    models = [
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "together_ai/mistralai/Mistral-Small-24B-Instruct-2501",
    ]
    question = "時系列分析とは何ですか？一文で簡潔に答えてください。"

    print("=== Together AI 上の複数モデル比較 ===\n")
    results = await asyncio.gather(
        *[call_model(m, question) for m in models]
    )

    for r in results:
        short_name = r["model"].split("/")[-1]
        if "error" in r:
            print(f"[{short_name}] エラー: {r['error']}\n")
        else:
            print(f"[{short_name}] ({r['time']:.2f}秒)")
            print(f"  回答: {r['response']}")
            print(f"  トークン数: {r['tokens']}\n")


asyncio.run(main())
