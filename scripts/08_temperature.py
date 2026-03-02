# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""08_temperature.py - temperatureパラメータの効果"""

from litellm import completion

prompt = "AIの未来について一文で予測してください。"
temperatures = [0.0, 0.5, 1.0, 1.5]

for temp in temperatures:
    print(f"=== temperature = {temp} ===")
    # 同じtemperatureで3回生成して、ばらつきを確認
    for i in range(3):
        response = completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100,
        )
        print(f"  [{i+1}] {response.choices[0].message.content}")
    print()
