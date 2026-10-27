"""Agent 核心：基于 DeepSeek Function Calling 的任务执行链路（流式版）

流程：意图识别（模型自动选择工具）→ 工具调用 → 结果整合 → 继续或流式生成回答。
- run_agent_stream: 生成器，逐段产出事件（text / tool_round / done），供 UI 流式渲染
- run_agent:      非流式封装，收集完整回答，供命令行与评估脚本使用
多轮对话由调用方维护 history（仅保留 user / assistant 文本轮次）。
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS_SCHEMA, execute_tool
from memory import load_facts

load_dotenv()

SYSTEM_PROMPT = (
    "你是安徽大学的校园生活助理。回答规则：\n"
    "1. 涉及实时信息（时间、天气）、数值计算、校规校纪、个人信息（课表/成绩/校园卡）的问题，"
    "必须调用对应工具获取结果，不要凭记忆编造；\n"
    "2. 可以在一次回复中组合调用多个工具；\n"
    "3. 调用工具时不要输出多余文字，等拿到工具结果后再组织回答；\n"
    "4. 工具返回 {\"ok\": false} 时，向用户说明失败原因并给出建议，不要编造结果；\n"
    "5. 校园知识库的回答要注明引用片段；\n"
    "6. 与工具无关的日常问题直接回答，用简洁的中文。"
)

MAX_ROUNDS = 5  # 单次提问最多经历的"模型→工具"轮数，防止死循环


def _get_client():
    return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def _build_messages(user_message: str, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    facts = load_facts()
    if facts:
        messages[0]["content"] += "\n\n已知的用户偏好（回答时可自然利用）：\n- " + "\n- ".join(facts)
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


def run_agent_stream(user_message: str, history=None, client=None):
    """流式执行一轮 Agent 任务。

    依次产出事件：
      {"type": "text", "delta": str}              最终回答的增量文本
      {"type": "tool_round", "records": [...]}    一轮工具调用记录
      {"type": "done", "answer": str}             结束，完整回答
    """
    client = client or _get_client()
    history = history or []
    messages = _build_messages(user_message, history)
    all_records = []

    for _ in range(MAX_ROUNDS):
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS_SCHEMA,
            stream=True,
            temperature=0.3,
        )

        content = ""
        tool_calls_acc = {}  # index -> {"id","name","arguments"}，流式增量累积
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta and delta.content:
                content += delta.content
                yield {"type": "text", "delta": delta.content}
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] += tc.id
                    if tc.function and tc.function.name:
                        acc["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments

        # 没有工具调用 → 本次内容即最终回答（已在上面流式产出）
        if not tool_calls_acc:
            yield {"type": "done", "answer": content or "（模型未返回内容，请重试）"}
            return

        # 有工具调用 → 执行并把结果回填给模型，进入下一轮
        messages.append({"role": "assistant", "content": content,
                         "tool_calls": [{"id": a["id"], "type": "function",
                                         "function": {"name": a["name"], "arguments": a["arguments"]}}
                                        for a in tool_calls_acc.values()]})
        round_records = []
        for acc in tool_calls_acc.values():
            try:
                args = json.loads(acc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(acc["name"], args)
            record = {"name": acc["name"], "args": args, "result": result}
            round_records.append(record)
            all_records.append(record)
            messages.append({"role": "tool", "tool_call_id": acc["id"], "content": result})
        yield {"type": "tool_round", "records": round_records}

    # 超过最大轮数仍没给出答案：兜底拒答，不编造
    fallback = "这个问题处理起来比较复杂，暂时没能完成。你可以换个问法，或直接咨询学校相关部门。"
    yield {"type": "text", "delta": fallback}
    yield {"type": "done", "answer": fallback}


def run_agent(user_message: str, history=None, client=None):
    """非流式封装：返回 (final_answer, tool_records)"""
    answer, records = "", []
    for event in run_agent_stream(user_message, history, client):
        if event["type"] == "tool_round":
            records.extend(event["records"])
        elif event["type"] == "done":
            answer = event["answer"]
    return answer, records


# ---------- 命令行入口：python agent.py ----------
if __name__ == "__main__":
    from memory import update_memory

    print("=== 校园生活助理 Agent（输入 exit 退出）===")
    history = []
    while True:
        q = input("你：").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break
        answer, records = "", []
        for event in run_agent_stream(q, history):
            if event["type"] == "tool_round":
                records.extend(event["records"])
                for r in event["records"]:
                    print(f"  🔧 {r['name']}({r['args']}) → {r['result'][:80]}")
            elif event["type"] == "text":
                answer += event["delta"]
                print(event["delta"], end="", flush=True)
        print("\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        update_memory(q, answer)  # 沉淀用户偏好
