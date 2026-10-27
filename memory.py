"""长期记忆：从对话中抽取用户偏好，持久化到 user_memory.json，跨会话生效

设计要点：
- 抽取用一次轻量的 DeepSeek 调用完成，任何失败都静默跳过，绝不影响主对话；
- 事实以短句形式存储，最多保留 20 条，新旧按时间戳排序；
- 只记录"稳定的偏好/信息"（如常用城市、姓名、年级），不记录一次性问答内容。
"""
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MEMORY_FILE = "user_memory.json"
MAX_FACTS = 20

_EXTRACT_PROMPT = (
    "从下面这段用户与助手的对话中，抽取值得长期记住的用户偏好或个人信息"
    "（例如：所在城市、常用查询、姓名、身份、习惯等）。\n"
    "规则：\n"
    "- 只输出 JSON 数组，每项一句短中文；没有值得记的就输出 []\n"
    "- 不要记录一次性的问答内容、数字结果或客套话\n"
    "- 最多 3 条\n\n"
    "对话：\n{dialogue}"
)


def load_facts() -> list:
    """读取当前记忆的事实列表（供注入 System Prompt）"""
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return [item["fact"] for item in json.load(f).get("facts", [])]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def update_memory(user_message: str, assistant_reply: str):
    """每轮对话后调用：抽取新的偏好并合并进记忆。失败静默，不抛异常。"""
    try:
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
                dialogue=f"用户：{user_message}\n助手：{assistant_reply}")}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text[text.index("["): text.rindex("]") + 1]  # 容错：截取 JSON 数组部分
        new_facts = json.loads(text)
        if not isinstance(new_facts, list):
            return

        existing = load_facts()
        merged, seen = [], set()
        for fact in new_facts + existing:  # 新事实排前面
            fact = str(fact).strip()
            if fact and fact not in seen:
                seen.add(fact)
                merged.append({"fact": fact, "ts": datetime.now().isoformat(timespec="seconds")})

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"facts": merged[:MAX_FACTS]}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 记忆失败不影响主流程


def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
