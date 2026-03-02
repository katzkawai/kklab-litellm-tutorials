# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "litellm",
# ]
# ///
"""16_custom_callback.py - カスタムコールバックによるログ記録"""

import json
import time

import litellm
from litellm import completion
from litellm.integrations.custom_logger import CustomLogger


class SimpleLogger(CustomLogger):
    """API呼び出しのログを記録するカスタムロガー"""

    def __init__(self):
        self.logs = []

    def log_pre_api_call(self, model, messages, kwargs):
        print(f"[LOG] API呼び出し開始: model={model}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        elapsed = end_time - start_time
        entry = {
            "model": kwargs.get("model", "unknown"),
            "elapsed_seconds": elapsed.total_seconds(),
            "prompt_tokens": getattr(
                response_obj.usage, "prompt_tokens", 0
            ),
            "completion_tokens": getattr(
                response_obj.usage, "completion_tokens", 0
            ),
            "total_tokens": getattr(
                response_obj.usage, "total_tokens", 0
            ),
        }
        self.logs.append(entry)
        print(
            f"[LOG] 成功: {entry['model']} "
            f"({entry['elapsed_seconds']:.2f}秒, "
            f"{entry['total_tokens']}トークン)"
        )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"[LOG] 失敗: {kwargs.get('model', 'unknown')}")


# ロガーを登録
logger = SimpleLogger()
litellm.callbacks = [logger]

# 複数回のAPI呼び出し
models = ["openai/gpt-4o-mini", "openai/gpt-4o-mini"]
for model in models:
    response = completion(
        model=model,
        messages=[{"role": "user", "content": "1+1=?"}],
        max_tokens=50,
    )

# ログのサマリーを表示
print("\n=== ログサマリー ===")
print(json.dumps(logger.logs, indent=2, ensure_ascii=False))
