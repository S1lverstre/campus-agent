"""工具插件层：天气查询 / 日期时间 / 数学计算 / 校园知识库问答

每个工具返回统一的 JSON 字符串：{"ok": true, "data": ...} 或 {"ok": false, "error": ...}，
供 Agent 层统一处理成功/失败分支。
"""
import ast
import json
import operator
import re
from datetime import datetime

# ---------- 工具 1：日期时间 ----------

def get_current_datetime() -> str:
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return json.dumps({
        "ok": True,
        "data": {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": weekdays[now.weekday()],
        },
    }, ensure_ascii=False)


# ---------- 工具 2：数学计算 ----------

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node):
    """只允许四则运算/幂/取模的递归求值，杜绝任意代码执行"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("仅支持加减乘除、幂和取模运算")


def calculate(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        result = round(result, 6)
        return json.dumps({"ok": True, "data": {"expression": expression, "result": result}},
                          ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"计算失败：{e}"}, ensure_ascii=False)


# ---------- 工具 3：天气查询（Open-Meteo 免费API，无需密钥） ----------

import urllib.request
import urllib.parse

_CITY_COORDS = {  # 常用城市内置坐标，查不到再走地理编码接口
    "合肥": (31.82, 117.23), "北京": (39.90, 116.41), "上海": (31.23, 121.47),
    "南京": (32.06, 118.80), "杭州": (30.27, 120.16), "武汉": (30.59, 114.31),
    "广州": (23.13, 113.26), "深圳": (22.54, 114.06), "成都": (30.57, 104.07),
    "西安": (34.34, 108.94), "蚌埠": (32.92, 117.39), "芜湖": (31.35, 118.38),
}

_WMO_CODE = {0: "晴", 1: "多云", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
             51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "中雨",
             65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨",
             81: "阵雨", 82: "强阵雨", 95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹"}


def _geocode(city: str):
    """Open-Meteo 地理编码：城市名 → 经纬度"""
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "zh", "format": "json"})
    with urllib.request.urlopen(url, timeout=8) as resp:
        results = json.loads(resp.read().decode("utf-8")).get("results", [])
    if not results:
        raise ValueError(f"未找到城市：{city}")
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city)


def get_weather(city: str) -> str:
    last_err = None
    for attempt in range(3):  # 网络请求失败自动重试，最多3次
        try:
            coords = _CITY_COORDS.get(city)
            if coords:
                lat, lon = coords
                name = city
            else:
                lat, lon, name = _geocode(city)
            url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon, "current_weather": "true",
                "timezone": "Asia/Shanghai"})
            with urllib.request.urlopen(url, timeout=8) as resp:
                cw = json.loads(resp.read().decode("utf-8"))["current_weather"]
            return json.dumps({"ok": True, "data": {
                "city": name,
                "temperature": cw["temperature"],
                "windspeed": cw["windspeed"],
                "weather": _WMO_CODE.get(cw["weathercode"], "未知"),
            }}, ensure_ascii=False)
        except Exception as e:
            last_err = e
    return json.dumps({"ok": False, "error": f"天气查询失败（已重试3次）：{last_err}"},
                      ensure_ascii=False)


# ---------- 工具 4：校园知识库问答（复用 RAG 检索链路） ----------

def campus_kb_qa(question: str) -> str:
    from rag import answer_from_kb  # 延迟导入，避免未装 RAG 依赖时其他工具不可用
    try:
        return json.dumps({"ok": True, "data": answer_from_kb(question)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"知识库问答失败：{e}"}, ensure_ascii=False)


# ---------- 工具 5：联网搜索（Bing 网页结果，零第三方依赖） ----------

def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").replace("&ensp;", " ").replace("&#0183;", "·").strip()


def web_search(query: str) -> str:
    """Bing 网页搜索，返回前 5 条 标题/链接/摘要。知识库覆盖不到的时效性问题走这里"""
    last_err = None
    for attempt in range(3):
        try:
            url = "https://www.bing.com/search?" + urllib.parse.urlencode(
                {"q": query, "mkt": "zh-CN", "setlang": "zh-CN"})
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode("utf-8", "ignore")
            results = []
            for block in re.findall(r'<li class="b_algo"(.*?)</li>', html, re.S)[:5]:
                title = re.search(r"<h2[^>]*>.*?<a[^>]*>(.*?)</a>", block, re.S)
                link = re.search(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"', block, re.S)
                snippet = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
                if title and link:
                    results.append({
                        "title": _strip_tags(title.group(1)),
                        "url": link.group(1),
                        "snippet": _strip_tags(snippet.group(1) if snippet else "")[:150],
                    })
            if not results:
                raise ValueError("未解析到搜索结果")
            return json.dumps({"ok": True, "data": {"query": query, "results": results}},
                              ensure_ascii=False)
        except Exception as e:
            last_err = e
    return json.dumps({"ok": False, "error": f"联网搜索失败（已重试3次）：{last_err}"},
                      ensure_ascii=False)


# ---------- 工具 6-8：校园服务（模拟数据，演示多工具编排） ----------

def _load_mock() -> dict:
    with open("mock_user.json", encoding="utf-8") as f:
        return json.load(f)


def campus_card_balance() -> str:
    """校园卡余额查询（模拟数据）"""
    card = _load_mock()["campus_card"]
    return json.dumps({"ok": True, "data": {
        "balance": card["balance"],
        "last_expense": f"{card['last_expense']['time']} 在{card['last_expense']['place']}消费 {card['last_expense']['amount']} 元",
    }}, ensure_ascii=False)


def get_week_schedule() -> str:
    """本周课表查询（模拟数据）"""
    return json.dumps({"ok": True, "data": {"schedule": _load_mock()["week_schedule"]}},
                      ensure_ascii=False)


def query_grades(semester: str = "") -> str:
    """成绩与绩点查询（模拟数据）"""
    g = _load_mock()["grades"]
    return json.dumps({"ok": True, "data": {
        "semester": semester or "2025-2026学年",
        "gpa": g["gpa"], "ranking": g["ranking"], "courses": g["courses"],
    }}, ensure_ascii=False)


# ---------- Function Calling 的工具 schema（供 DeepSeek 选择） ----------

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_current_datetime",
        "description": "获取当前的日期、时间和星期。任何与'今天几号/现在几点/星期几'有关的问题都必须调用此工具",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "精确计算数学表达式，支持加减乘除、幂(^写作**)和取模。凡是需要数值计算的场合都应调用，不要心算",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Python 语法的数学表达式，如 (3+5)*12/2"},
        }, "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询中国城市的实时天气（温度、风向风速、天气现象）",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，如：合肥"},
        }, "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "campus_kb_qa",
        "description": "查询安徽大学规章制度知识库：转专业、奖学金、学籍管理、宿舍管理、校园卡等。凡涉及校规校纪的问题必须调用此工具",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "用户的问题原文"},
        }, "required": ["question"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "联网搜索最新信息。当校园知识库、天气工具都覆盖不到，或问题涉及时效性新闻、常识百科时调用",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "campus_card_balance",
        "description": "查询用户校园卡的当前余额和最近一笔消费记录",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_week_schedule",
        "description": "查询用户本周的课程表（每天有哪些课、时间和教室）",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "query_grades",
        "description": "查询用户的成绩、绩点（GPA）与专业排名",
        "parameters": {"type": "object", "properties": {
            "semester": {"type": "string", "description": "学期，如 2025-2026学年，可留空"},
        }, "required": []},
    }},
]


def execute_tool(name: str, args: dict) -> str:
    """统一入口：分发工具调用并兜底，任何异常都转成 {"ok": false} 而不是让 Agent 崩溃"""
    dispatch = {
        "get_current_datetime": lambda a: get_current_datetime(),
        "calculate": lambda a: calculate(a["expression"]),
        "get_weather": lambda a: get_weather(a["city"]),
        "campus_kb_qa": lambda a: campus_kb_qa(a["question"]),
        "web_search": lambda a: web_search(a["query"]),
        "campus_card_balance": lambda a: campus_card_balance(),
        "get_week_schedule": lambda a: get_week_schedule(),
        "query_grades": lambda a: query_grades(a.get("semester", "")),
    }
    if name not in dispatch:
        return json.dumps({"ok": False, "error": f"未知工具：{name}"}, ensure_ascii=False)
    try:
        return dispatch[name](args)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"工具执行异常：{e}"}, ensure_ascii=False)
