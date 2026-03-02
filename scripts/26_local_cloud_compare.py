# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""26_local_cloud_compare.py - ローカルモデルとクラウドAPIの比較

ローカル推論エンジン（Ollama, vLLM等）とクラウドAPIの応答を
同じインターフェースで比較する例。

事前に Ollama でモデルを取得しておくこと:
  ollama pull llama3.2:3b
"""

import asyncio
import time

from litellm import acompletion


async def call_model(model: str, question: str, **kwargs) -> dict:
    """モデルを呼び出して応答と計測結果を返す"""
    start = time.time()
    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=200,
            **kwargs,
        )
        elapsed = time.time() - start
        return {
            "model": model,
            "response": response.choices[0].message.content,
            "tokens": response.usage.total_tokens,
            "time": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "model": model,
            "response": None,
            "tokens": 0,
            "time": elapsed,
            "error": str(e),
        }


async def main():
    question = "重回帰分析と単回帰分析の違いを一文で説明してください。"

    # ローカル (Ollama) とクラウド (OpenAI) を並列で呼び出し
    tasks = [
        call_model(
            "ollama_chat/llama3.2:3b",
            question,
            api_base="http://localhost:11434",
        ),
        call_model("openai/gpt-4o-mini", question),
    ]

    print("=== ローカル vs クラウド 比較 ===\n")
    results = await asyncio.gather(*tasks)

    for r in results:
        tag = "ローカル" if "ollama" in r["model"] else "クラウド"
        print(f"[{tag}] {r['model']} ({r['time']:.2f}秒)")
        if r["error"]:
            print(f"  エラー: {r['error']}")
        else:
            print(f"  回答: {r['response']}")
            print(f"  トークン数: {r['tokens']}")
        print()


asyncio.run(main())
