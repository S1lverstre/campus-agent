"""Streamlit 对话界面：流式输出 + 工具调用可视化 + 会话统计 + 长期记忆 + 导出"""
import json
import time
from datetime import datetime

import streamlit as st
from agent import run_agent_stream
from memory import load_facts, clear_memory, update_memory

st.set_page_config(page_title="校园生活助理 Agent", page_icon="🎓")
st.title("🎓 校园生活助理 Agent")
st.caption("DeepSeek Function Calling ｜ 8 个工具：日期时间 · 计算 · 天气 · 知识库问答 · 联网搜索 · 校园卡 · 课表 · 成绩")

if "messages" not in st.session_state:  # [{role, content, tool_records}]
    st.session_state.messages = []
if "tool_stats" not in st.session_state:  # {工具名: 次数}
    st.session_state.tool_stats = {}

TOOL_ICONS = {
    "get_current_datetime": "🕐", "calculate": "🧮", "get_weather": "🌤️",
    "campus_kb_qa": "📚", "web_search": "🔍", "campus_card_balance": "💳",
    "get_week_schedule": "📅", "query_grades": "📊",
}

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("🧰 已注册工具（8 个）")
    if st.session_state.tool_stats:
        st.subheader("📈 本次会话工具调用统计")
        for name, count in sorted(st.session_state.tool_stats.items(), key=lambda x: -x[1]):
            st.progress(min(count / 10, 1.0), text=f"{TOOL_ICONS.get(name, '🔧')} {name} × {count}")
    else:
        st.caption("对话后这里会显示各工具的调用次数")

    st.divider()
    st.subheader("🧠 长期记忆")
    facts = load_facts()
    if facts:
        for fact in facts:
            st.markdown(f"- {fact}")
        if st.button("🗑️ 清空记忆"):
            clear_memory()
            st.rerun()
    else:
        st.caption("暂无记忆。聊聊你的偏好（如“我常用合肥的天气”），我会记住并在下次启动时生效")

    st.divider()
    if st.session_state.messages and st.button("💾 导出对话为 Markdown"):
        lines = [f"# 与校园生活助理的对话（{datetime.now():%Y-%m-%d %H:%M}）\n"]
        for m in st.session_state.messages:
            lines.append(f"**{'🙋 用户' if m['role'] == 'user' else '🤖 助理'}**：{m['content']}\n")
            for r in m.get("tool_records", []):
                lines.append(f"> 🔧 `{r['name']}` `{json.dumps(r['args'], ensure_ascii=False)}`\n>")
                lines.append(f"> ```json\n> {r['result'][:300]}\n> ```\n")
        st.download_button("下载 markdown", "\n".join(lines),
                           file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.md",
                           mime="text/markdown", key="dl")

# ---------- 渲染历史消息 ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("tool_records"):
            with st.expander(f"🔧 调用了 {len(m['tool_records'])} 个工具", expanded=False):
                for i, r in enumerate(m["tool_records"], 1):
                    st.markdown(f"**{i}. {TOOL_ICONS.get(r['name'], '🔧')} `{r['name']}`**  参数：`{r['args']}`")
                    st.code(r["result"], language="json")

# ---------- 用户输入 ----------
if prompt := st.chat_input("试试：现在几点？/ 我周三有什么课？/ 卡里还有多少钱？/ 搜一下安大新闻"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages if m["role"] != "system"][:-1]

        answer, all_records = "", []
        stream_box = st.empty()          # 流式文本容器
        tool_box = st.empty()            # 工具调用过程容器

        try:
            for event in run_agent_stream(prompt, history):
                if event["type"] == "tool_round":
                    all_records.extend(event["records"])
                    for r in event["records"]:
                        st.session_state.tool_stats[r["name"]] = st.session_state.tool_stats.get(r["name"], 0) + 1
                    with tool_box.container():
                        with st.expander(f"🔧 已调用 {len(all_records)} 个工具", expanded=True):
                            for i, r in enumerate(all_records, 1):
                                st.markdown(f"**{i}. {TOOL_ICONS.get(r['name'], '🔧')} `{r['name']}`**  参数：`{r['args']}`")
                                st.code(r["result"], language="json")
                    stream_box.empty()  # 有工具轮次时清掉占位文本
                    stream_box = st.empty()
                elif event["type"] == "text":
                    answer += event["delta"]
                    stream_box.markdown(answer + " ▌")  # 打字机效果
                elif event["type"] == "done":
                    answer = event["answer"]
        except Exception as e:
            stream_box.error(
                "⚠️ 当前为界面演示模式（服务端未配置 API Key），无法调用大模型。\n\n"
                "本项目完整功能需在 `.env` 中配置 `DEEPSEEK_API_KEY` 与 `SILICONFLOW_API_KEY` 后本地运行。"
            )
            answer = answer or "（演示模式，未能生成回答）"

        stream_box.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_records": all_records})

    # 后台沉淀长期记忆（失败静默，不影响界面）
    update_memory(prompt, answer)
