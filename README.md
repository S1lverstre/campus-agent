# 基于 Function Calling 的校园生活助理 Agent

[![CI](https://github.com/S1lverstre/campus-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/S1lverstre/campus-agent/actions/workflows/ci.yml)

安徽大学校园生活助理：在 campus-rag 知识库问答的基础上，引入 **DeepSeek Function Calling**，
模型自动判断意图并从 **8 个工具**中选择调用——日期时间、数学计算、实时天气、校园知识库问答（复用 RAG 检索链路）、
联网搜索、校园卡余额、课表查询、成绩查询，支持一次回复组合调用多个工具，并具备**长期记忆**能力。

## 架构

```
用户提问
   │
   ▼
Agent 主循环（agent.py，流式）──────── DeepSeek chat API
   │  意图识别 → 选择工具 → 抽取参数（最多 5 轮，超限兜底拒答）
   ▼
工具层（tools.py）—— 统一返回 {"ok": bool, ...}
   ├─ get_current_datetime   日期时间
   ├─ calculate              安全表达式求值（AST 白名单）
   ├─ get_weather            Open-Meteo 实时天气（免费API，失败自动重试3次）
   ├─ campus_kb_qa           知识库问答（rag.py：FAISS Top-5 检索 + DeepSeek 生成）
   ├─ web_search             联网搜索（Bing 网页解析，零第三方依赖，重试3次）
   ├─ campus_card_balance    校园卡余额（模拟数据）
   ├─ get_week_schedule      本周课表（模拟数据）
   └─ query_grades           成绩绩点（模拟数据）
   │
   ▼
结果整合 → 继续调用工具 / 流式生成最终回答
   │
   ▼
长期记忆（memory.py）：每轮用轻量 LLM 调用抽取用户偏好 → user_memory.json → 注入下次 System Prompt
```

## 运行

```bash
pip install -r requirements.txt
# .env 中配置 DEEPSEEK_API_KEY 与 SILICONFLOW_API_KEY

python agent.py                      # 命令行对话（流式输出）
streamlit run app.py                 # 网页版（流式 + 工具可视化 + 会话统计 + 记忆面板 + 导出）

python evaluate.py                   # 工具层评估 12 例（离线）
python evaluate.py --with-agent      # 加跑 Agent 端到端 9 例（消耗API额度）
python build_index.py                # 如需重建知识库索引
```

Docker 一键运行：

```bash
docker build -t campus-agent .
docker run -p 8501:8501 --env-file .env campus-agent
```

## 设计要点

- **意图识别与工具选择交给模型**：8 个工具以 JSON Schema 注册，DeepSeek 按语义自动选择，
  支持一次回复组合调用多个工具；
- **统一工具返回协议**：所有工具返回 `{"ok": true/false, ...}`，失败时模型据此向用户解释，
  而不是编造结果；
- **网络重试与兜底**：天气/搜索失败自动重试 3 次；Agent 主循环最多 5 轮，超限拒答；
- **计算工具安全求值**：基于 AST 白名单，只允许四则/幂/取模，杜绝任意代码执行；
- **RAG 复用**：知识库问答工具复用 campus-rag 的 FAISS 索引（bge-m3 向量 + Top-5 检索 + 引用溯源）；
- **流式输出**：OpenAI SDK 流式接口 + 工具调用增量累积，打字机效果实时呈现；
- **长期记忆**：每轮对话后用轻量 LLM 调用抽取稳定偏好（静默失败），持久化后跨会话注入 System Prompt。
