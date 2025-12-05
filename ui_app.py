"""
Streamlit Web UI for a minimal LangChain RAG demo with guardrails:
- 侧边栏上传/删除知识库文件
- 风险判定、上下文净化、输出判审、canary 检测、会话暴露预算
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "amazon/nova-2-lite-v1:free"
CANARY_DEFAULT = "CANARY"
REQUEST_TIMEOUT = 30
SESSION_BUDGET_LIMIT = 1500  # max chars to return per session before refusal


def load_documents() -> List[Document]:
    """Load plain text documents from the data directory and chunk them."""
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": path.name, "chunk": i},
                )
            )
    return docs


def list_text_files() -> List[Path]:
    """Return sorted .txt files under the data directory."""
    return sorted(DATA_DIR.glob("*.txt"))


def write_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded txt file and return its path."""
    name = Path(uploaded_file.name).name or "upload.txt"
    target = DATA_DIR / name
    if target.exists():
        stem = target.stem
        suffix = target.suffix or ".txt"
        counter = 1
        while True:
            candidate = DATA_DIR / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            counter += 1
    target.write_text(uploaded_file.getvalue().decode("utf-8", errors="ignore"), encoding="utf-8")
    return target


def score_risk(text: str) -> int:
    """Heuristic risk score: higher means more likely exfil intent."""
    risky_keywords = [
        "输出全文",
        "全部内容",
        "不要省略",
        "逐字",
        "打印",
        "泄露",
        "继续上一段",
        "下一段",
        "所有数据",
        "完整文本",
        "ignore previous",
        "verbatim",
    ]
    score = sum(1 for kw in risky_keywords if kw.lower() in text.lower())
    return score


def scrub_text(text: str) -> str:
    """Remove imperative injection hints from context."""
    patterns = [
        r"(?i)ignore .*",
        r"(?i)disregard .*",
        r"(?i)请忽略.*",
        r"(?i)遵循.*以下.*指令",
    ]
    clean = text
    for pat in patterns:
        clean = re.sub(pat, "", clean)
    return clean.strip()


def detect_canary(text: str, canary: str) -> bool:
    return canary and canary.lower() in text.lower()


@st.cache_resource(show_spinner=False)
def build_components(model_name: str, top_k: int):
    docs = load_documents()
    if not docs:
        return None, None

    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    headers: Dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_title = os.getenv("OPENROUTER_APP_TITLE")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_title:
        headers["X-Title"] = app_title

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
        timeout=REQUEST_TIMEOUT,
        max_retries=1,
    )

    return retriever, llm


def build_prompt(context: str, question: str, canary: str) -> List:
    sys = (
        "你是安全合规的助手，必须避免泄露敏感或内部信息，"
        "不得逐字输出原文。如发现敏感标记（如 CANARY 或密钥），拒绝输出并提醒安全风险。"
        "请用中文，摘要回答，并标注来源。"
    )
    user = (
        f"问题：{question}\n"
        f"上下文（已净化）：\n{context}\n"
        "请概括回答，避免长段原文，引用来源 ID。"
    )
    return [SystemMessage(content=sys), HumanMessage(content=user)]


def judge_answer(llm: ChatOpenAI, question: str, answer: str, canary: str, sources: List[Document]) -> str:
    """Return verdict: ok | leak | refuse | error."""
    src_names = ", ".join({doc.metadata.get("source", "unknown") for doc in sources})
    prompt = [
        SystemMessage(
            content=(
                "你是安全审查员。判定回答是否泄露敏感/原文/标记。"
                "只返回一个标签：ok / leak / refuse。"
            )
        ),
        HumanMessage(
            content=(
                f"问题: {question}\n"
                f"回答: {answer}\n"
                f"来源: {src_names}\n"
                f"敏感标记: {canary or '无'}"
            )
        ),
    ]
    try:
        verdict = llm.invoke(prompt).content.strip().lower()
        if "leak" in verdict or "泄" in verdict:
            return "leak"
        if "refuse" in verdict or "拒" in verdict:
            return "refuse"
        return "ok"
    except Exception:
        return "error"


def main() -> None:
    st.set_page_config(page_title="LangChain RAG Web UI", layout="wide")
    st.title("LangChain RAG Web UI")
    st.markdown(
        "- 将文本文件放入 `./data` 目录（默认附带一个 sample）。\n"
        "- 设置 `OPENROUTER_API_KEY`（必要），可选 `OPENROUTER_SITE_URL` 与 `OPENROUTER_APP_TITLE` 用于标注请求。\n"
        "- 运行：`streamlit run ui_app.py`，可调整模型名与 Top-K 检索范围后提问。"
    )

    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning("未检测到 OPENROUTER_API_KEY，需设置后才能调用 OpenRouter 模型。", icon="⚠️")

    with st.sidebar:
        st.header("配置")
        model_name = st.text_input("模型名称", value=DEFAULT_MODEL)
        top_k = st.slider("检索 Top-K", min_value=1, max_value=8, value=4, step=1)
        canary_token = st.text_input("Canary 标记", value=CANARY_DEFAULT)
        budget_limit = st.slider("会话输出预算(字符)", min_value=500, max_value=5000, value=SESSION_BUDGET_LIMIT, step=100)

        st.divider()
        st.header("知识库管理")
        uploaded_files = st.file_uploader("上传 .txt 文件", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            saved = []
            for f in uploaded_files:
                path = write_uploaded_file(f)
                saved.append(path.name)
            build_components.clear()
            st.success(f"已上传: {', '.join(saved)}")
            st.button("应用并重载", type="primary", on_click=st.experimental_rerun)

        existing_files = [p.name for p in list_text_files()]
        to_delete = st.multiselect("选择要删除的文件", options=existing_files)
        if st.button("删除选中", disabled=not to_delete):
            for name in to_delete:
                (DATA_DIR / name).unlink(missing_ok=True)
            build_components.clear()
            st.success(f"已删除: {', '.join(to_delete)}")
            st.button("应用并重载", type="primary", on_click=st.experimental_rerun)

        if st.button("重新加载知识库", type="primary"):
            build_components.clear()
            st.experimental_rerun()

    retriever, llm = build_components(model_name=model_name, top_k=top_k)
    if retriever is None or llm is None:
        st.info("`data/` 下暂无 .txt 文件，请添加后点击侧边栏的“重新加载知识库”。")
        return

    # 会话预算
    if "budget_used" not in st.session_state:
        st.session_state.budget_used = 0

    question = st.text_input("请输入问题", placeholder="例如：这份知识库的核心内容是什么？")
    if not question:
        return

    risk = score_risk(question)
    st.write(f"风险评分: {risk}")
    if st.session_state.budget_used >= budget_limit:
        st.error("会话输出预算已用尽，避免继续输出以防泄露。")
        return

    with st.spinner("生成中..."):
        try:
            raw_docs = retriever.get_relevant_documents(question)
            cleaned_docs = []
            for d in raw_docs:
                cleaned = d.copy()
                cleaned.page_content = scrub_text(cleaned.page_content)
                cleaned_docs.append(cleaned)

            # 构造上下文
            ctx_blocks = []
            for doc in cleaned_docs:
                meta = doc.metadata
                ctx_blocks.append(f"[{meta.get('source','unknown')}#{meta.get('chunk',0)}]\n{doc.page_content}")
            context = "\n\n".join(ctx_blocks) if ctx_blocks else "(无命中文档)"

            prompt = build_prompt(context=context, question=question, canary=canary_token)
            answer = llm.invoke(prompt).content

            verdict = judge_answer(llm, question, answer, canary_token, cleaned_docs)
            leaked = detect_canary(answer, canary_token) or any(detect_canary(doc.page_content, canary_token) for doc in cleaned_docs)

            if verdict in {"leak", "refuse"} or leaked:
                answer = "检测到可能的敏感信息或风险，已拒绝输出。"

            st.session_state.budget_used += len(answer)
            result = {"result": answer, "source_documents": cleaned_docs, "verdict": verdict, "leaked": leaked}
        except Exception as exc:  # broad: surface LLM/HTTP errors to UI
            st.error(f"调用模型失败：{exc}")
            return

    st.subheader("回答")
    st.write(result["result"])
    st.caption(f"判定: {result.get('verdict','?')} | Canary 泄露: {result.get('leaked')} | 会话已用: {st.session_state.budget_used}/{budget_limit}")

    st.subheader("检索来源")
    for doc in result["source_documents"]:
        meta = doc.metadata
        st.markdown(f"- **{meta.get('source', 'unknown')}** (chunk {meta.get('chunk', 0)})")
        st.code(doc.page_content, language="text")


if __name__ == "__main__":
    main()
