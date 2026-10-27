"""复用 campus-rag 的检索链路：FAISS 相似度检索 + DeepSeek 依据资料作答"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

_client = None
_db = None


def _get_db():
    global _db
    if _db is None:
        embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            check_embedding_ctx_length=False,
        )
        _db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return _db


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    return _client


def answer_from_kb(question: str) -> dict:
    """知识库问答：检索 Top-5 → 组装 Prompt → 生成带引用的回答"""
    db = _get_db()
    hits = db.similarity_search(question, k=5)
    context = "\n\n".join(f"[片段{i+1}] {h.page_content}" for i, h in enumerate(hits))
    prompt = (
        "你是安徽大学校园助手。请仅根据以下资料回答问题，"
        "回答末尾注明参考了哪些片段；如果资料中没有答案，就回答"
        "\"资料中未找到，建议咨询学校相关部门\"。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    resp = _get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {
        "answer": resp.choices[0].message.content,
        "references": [
            {"source": h.metadata.get("source", "?"), "snippet": h.page_content[:80]}
            for h in hits
        ],
    }
