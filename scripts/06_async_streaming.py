# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""06_async_streaming.py - 非同期ストリーミング応答"""

import asyncio

from litellm import acompletion


async def async_stream():
    """非同期ストリーミングの例"""
    print("=== 非同期ストリーミング ===")
    response = await acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "統計学の基本概念を3つ説明してください。"}
        ],
        stream=True,
    )

    async for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

    print()


asyncio.run(async_stream())
