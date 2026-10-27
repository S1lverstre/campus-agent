"""自动化评估（共 21 例）：
1. 工具层 12 例（离线，不调 API）：覆盖正常输入、边界输入、非法输入兜底
2. Agent 端到端 9 例（需 API key）：验证"问题 → 是否调对工具 → 是否组合调用 → 回答是否合理"
"""
import json
from tools import (calculate, get_current_datetime, campus_card_balance,
                   get_week_schedule, query_grades, execute_tool)

# ---------- 工具层用例（离线可跑） ----------
TOOL_CASES = [
    ("计算：四则混合", lambda: calculate("(3+5)*12/2"), lambda r: json.loads(r)["data"]["result"] == 48.0),
    ("计算：运算优先级", lambda: calculate("2+3*4"), lambda r: json.loads(r)["data"]["result"] == 14),
    ("计算：幂运算", lambda: calculate("2**10"), lambda r: json.loads(r)["data"]["result"] == 1024),
    ("计算：负数与浮点", lambda: calculate("-5/2"), lambda r: json.loads(r)["data"]["result"] == -2.5),
    ("计算：除零兜底", lambda: calculate("1/0"), lambda r: json.loads(r)["ok"] is False),
    ("计算：非法代码兜底", lambda: calculate("import os"), lambda r: json.loads(r)["ok"] is False),
    ("计算：非表达式兜底", lambda: calculate("abc"), lambda r: json.loads(r)["ok"] is False),
    ("日期时间：含日期与星期", lambda: get_current_datetime(),
     lambda r: "datetime" in json.loads(r)["data"] and "星期" in json.loads(r)["data"]["weekday"]),
    ("校园卡：余额与最近消费", lambda: campus_card_balance(),
     lambda r: "balance" in json.loads(r)["data"] and "消费" in json.loads(r)["data"]["last_expense"]),
    ("课表：包含课程信息", lambda: get_week_schedule(),
     lambda r: "schedule" in json.loads(r)["data"]),
    ("成绩：GPA 与课程列表", lambda: query_grades(),
     lambda r: json.loads(r)["data"]["gpa"] > 0 and len(json.loads(r)["data"]["courses"]) > 0),
    ("未知工具兜底", lambda: execute_tool("no_such_tool", {}),
     lambda r: json.loads(r)["ok"] is False and "未知工具" in json.loads(r)["error"]),
]


def test_tools():
    print("== 工具层评估 ==")
    passed = 0
    for name, fn, check in TOOL_CASES:
        try:
            ok = check(fn())
        except Exception as e:
            ok = False
            print(f"  [异常] {name}: {e}")
        print(f"  {'✅' if ok else '❌'} {name}")
        passed += ok
    print(f"通过 {passed}/{len(TOOL_CASES)}\n")
    return passed, len(TOOL_CASES)


# ---------- Agent 端到端用例（需 API） ----------
AGENT_CASES = [
    {"question": "现在几点了？", "expect_tools": ["get_current_datetime"]},
    {"question": "1024 乘以 768 再除以 3 等于多少？", "expect_tools": ["calculate"]},
    {"question": "合肥今天天气怎么样？", "expect_tools": ["get_weather"]},
    {"question": "安徽大学转专业有什么要求？", "expect_tools": ["campus_kb_qa"]},
    {"question": "现在几点了？合肥今天天气怎么样？",  # 复合问题：应组合调用两个工具
     "expect_tools": ["get_current_datetime", "get_weather"]},
    {"question": "帮我联网搜一下安徽大学最近有什么新闻", "expect_tools": ["web_search"]},
    {"question": "我的校园卡里还剩多少钱？", "expect_tools": ["campus_card_balance"]},
    {"question": "我这周三有什么课？", "expect_tools": ["get_week_schedule"]},
    {"question": "用一句话介绍你自己", "expect_tools": []},  # 不应调用工具
]


def test_agent():
    from agent import run_agent
    print("== Agent 端到端评估 ==")
    passed = 0
    for case in AGENT_CASES:
        answer, records = run_agent(case["question"])
        used = [r["name"] for r in records]
        ok = all(t in used for t in case["expect_tools"]) and len(answer) > 5
        print(f"  {'✅' if ok else '❌'} {case['question']}  调用工具：{used or '无'}")
        passed += ok
    print(f"通过 {passed}/{len(AGENT_CASES)}\n")
    return passed, len(AGENT_CASES)


if __name__ == "__main__":
    p1, t1 = test_tools()
    import sys
    if "--with-agent" in sys.argv:  # 加参数才跑端到端（消耗 API 额度）
        p2, t2 = test_agent()
        print(f"总计：工具层 {p1}/{t1}，Agent 层 {p2}/{t2}")
    else:
        print(f"总计：工具层 {p1}/{t1}（Agent 端到端请加 --with-agent）")
