# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""31_together_cost_compare.py - Together AIとクラウドAPIのコスト比較

同じ質問を Together AI（オープンソースモデル）と
OpenAI（商用モデル）に送り、コストを比較する。

事前に環境変数を設定しておくこと:
  export TOGETHERAI_API_KEY="..."
  export OPENAI_API_KEY="sk-..."
"""

from litellm import completion, completion_cost

models = [
    ("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "Together AI (Llama 3.1 8B)"),
    ("together_ai/deepseek-ai/DeepSeek-V3", "Together AI (DeepSeek V3)"),
    ("openai/gpt-4o-mini", "OpenAI (GPT-4o mini)"),
]

question = (
    "計量経済学における操作変数法について、"
    "その目的と基本的な手順を300字程度で説明してください。"
)

print("=== コスト比較: Together AI vs OpenAI ===\n")
total_cost = 0.0

for model, label in models:
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=400,
        )

        cost = completion_cost(completion_response=response)
        total_cost += cost

        print(f"[{label}]")
        print(f"  入力トークン: {response.usage.prompt_tokens}")
        print(f"  出力トークン: {response.usage.completion_tokens}")
        print(f"  コスト: ${cost:.6f}")
        print(f"  回答冒頭: {response.choices[0].message.content[:80]}...")
        print()

    except Exception as e:
        print(f"[{label}] エラー: {e}\n")

print(f"合計コスト: ${total_cost:.6f}")
