import streamlit as st
import asyncio
import sys
import io
import re
import os
from pipeline import ResearchPipeline


# -------------------- Page config --------------------
st.set_page_config(
    page_title="Research Agent",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Research Agent")


# -------------------- Session state --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "running" not in st.session_state:
    st.session_state.running = False

if "logs" not in st.session_state:
    st.session_state.logs = ""


# -------------------- Logger --------------------
class StreamlitLogger(io.StringIO):
    def write(self, txt):
        st.session_state.logs += txt


# -------------------- Helpers --------------------
def extract_sources(md_text: str):
    """Extract bullet URLs from a SOURCES section"""
    sources = []
    if "SOURCES" in md_text.upper():
        section = md_text.upper().split("SOURCES", 1)[1]
        urls = re.findall(r"https?://\S+", section)
        sources.extend(urls)
    return list(dict.fromkeys(sources))  # dedupe


# -------------------- Render chat history --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])

        else:  # assistant
            st.markdown(msg["final_answer"])

            if msg.get("sources"):
                st.markdown("#### 🔗 Sources")
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

            if msg.get("markdown"):
                with st.expander("📄 Full Research Report (Markdown)"):
                    st.markdown(msg["markdown"], unsafe_allow_html=True)

            if msg.get("logs"):
                with st.expander("🛠️ Thinking / Internal reasoning"):
                    st.text(msg["logs"])


# -------------------- Chat input --------------------
prompt = st.chat_input(
    "Ask a deep factual or speculative question…",
    disabled=st.session_state.running
)

if prompt:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    st.session_state.running = True
    st.session_state.logs = ""

    logger = StreamlitLogger()
    old_stdout = sys.stdout
    sys.stdout = logger

    with st.spinner("Thinking… running multi-pass research"):
        try:
            pipeline = ResearchPipeline()
            result = asyncio.run(pipeline.run(prompt))

            final_answer = result["final_answer"]

            # Load markdown report
            #md_path = result["judge_evaluation"]["run_id"] if False else None
            # md_path =  result["md_path"]
            # You already know path in pipeline → safer to re-read directly
            # Instead, reuse pipeline logic:
            # markdown already read before vector store → re-read here

            # TEMP SAFE OPTION: reuse logs to extract markdown
            # md_text = ""
            # for line in st.session_state.logs.splitlines():
            #     if line.endswith(".md"):
            #         try:
            #             with open(line.strip(), "r") as f:
            #                 md_text = f.read()
            #         except:
                        # pass

            md_text = ""
            md_path = result.get("markdown_path")

            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()


            sources = extract_sources(md_text)

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "final_answer": final_answer,
                "sources": sources,
                "markdown": md_text,
                "logs": st.session_state.logs,
            })

        finally:
            sys.stdout = old_stdout
            st.session_state.running = False

    st.rerun()



