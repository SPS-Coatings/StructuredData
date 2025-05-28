# integrated_streamlit_app.py
"""
Unified Streamlit app that combines:
1️⃣  AI Data Visualization Agent  
    • Together AI LLM + E2B sandbox for on-the-fly Python/plot execution  
2️⃣  Data Analyst Agent  
    • φ (phi) DuckDB SQL agent + PandasTools for analyst-style Q&A

Run with:  streamlit run integrated_streamlit_app.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import io
import csv
import sys
import json
import tempfile
import contextlib
import warnings
from typing import Optional, List, Any, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
import base64

# 1️⃣  AI Data Visualization Agent deps
from together import Together
from e2b_code_interpreter import Sandbox

# 2️⃣  Data Analyst Agent deps
from phi.agent.duckdb import DuckDbAgent
from phi.model.openai import OpenAIChat
from phi.tools.pandas import PandasTools

# ──────────────────────────────────────────────────────────────────────────────
# Global Settings & Constants
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
st.set_page_config(page_title="📊 Unified Data Agent", page_icon="📊", layout="centered")

# Regex pattern to capture python code blocks from LLM responses
PY_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  AI Data Visualization Agent – Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def _code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    """Run arbitrary Python `code` in an isolated E2B sandbox, return results."""
    with st.spinner("🛠️ Executing Python code in E2B sandbox…"):
        stdout_capture, stderr_capture = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec_out = e2b_code_interpreter.run_code(code)

        # Mirror sandbox stdout / stderr to Streamlit console for debugging
        if stderr_capture.getvalue():
            print("[E2B-STDERR]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[E2B-STDOUT]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec_out.error:
            st.error(f"Sandbox error: {exec_out.error}")
            return None
        return exec_out.results


def _extract_python(llm_response: str) -> str:
    """Return the first ```python ... ``` block or empty string."""
    match = PY_BLOCK.search(llm_response)
    return match.group(1) if match else ""


def _chat_with_llm(
    e2b_code_interpreter: Sandbox,
    user_message: str,
    dataset_path: str,
    tg_api_key: str,
    model_name: str,
) -> Tuple[Optional[List[Any]], str]:
    """Forward user query + dataset location to Together-AI LLM, run returned code."""
    system_prompt = (
        "You are a senior Python data-scientist and data-visualization expert. "
        f"A CSV dataset lives at **{dataset_path}**. "
        "Answer the user strictly by:\n"
        "1. Thinking step-by-step.\n"
        "2. Emitting a single ```python …``` block that uses the dataset path above.\n"
        "3. After the code block, add a short plain-English explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner("🤖 LLM is thinking…"):
        client = Together(api_key=tg_api_key)
        resp = client.chat.completions.create(model=model_name, messages=messages)

    llm_msg = resp.choices[0].message.content
    python_code = _extract_python(llm_msg)

    if python_code:
        exec_results = _code_interpret(e2b_code_interpreter, python_code)
        return exec_results, llm_msg
    else:
        st.warning("No Python block found in the model output.")
        return None, llm_msg


def _upload_to_sandbox(e2b_sandbox: Sandbox, uploaded_file) -> str:
    """Persist uploaded file into sandbox FS; return in-sandbox path."""
    path_in_sandbox = f"./{uploaded_file.name}"
    try:
        e2b_sandbox.files.write(path_in_sandbox, uploaded_file)
        return path_in_sandbox
    except Exception as err:
        st.error(f"File upload to sandbox failed: {err}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  Data Analyst Agent – Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def _preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Clean a CSV/XLSX, coerce dates/numerics, save to temp CSV and return:
    (temp_path, list-of-columns, cleaned DataFrame)
    """
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            st.error("Unsupported format – upload CSV or Excel.")
            return None, None, None

        # Quote escaping
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # Attempt date / numeric coercion
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # Save to temp CSV (quoted)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df

    except Exception as exc:
        st.error(f"Pre-processing failed: {exc}")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – collect all API keys once
with st.sidebar:
    st.header("🔑 API Keys")
    tg_key = st.text_input("Together AI key (Data-Viz)", type="password")
    e2b_key = st.text_input("E2B key (Sandbox)", type="password")
    openai_key = st.text_input("OpenAI key (Data-Analyst)", type="password")

    # LLM dropdown for Data-Viz agent
    st.markdown("---")
    st.markdown("### Model for Data-Viz Agent")
    MODEL_OPTIONS = {
        "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "DeepSeek V3":        "deepseek-ai/DeepSeek-V3",
        "Qwen 2.5 7B":        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }
    chosen_model_human = st.selectbox("Choose model:", list(MODEL_OPTIONS.keys()), index=0)
    chosen_model_id = MODEL_OPTIONS[chosen_model_human]

    # Persist in session_state
    st.session_state["tg_key"] = tg_key
    st.session_state["e2b_key"] = e2b_key
    st.session_state["openai_key"] = openai_key
    st.session_state["model_id"] = chosen_model_id

st.title("📊 Unified Data Agent")

# Two-tab interface
viz_tab, analyst_tab = st.tabs(
    ["🖼️ AI Data Visualization Agent", "🗄️ Data Analyst Agent"]
)

# ────────────────────────────────
# Tab 1 – AI Data Visualization
# ────────────────────────────────
with viz_tab:
    st.subheader("AI Data Visualization Agent")
    st.markdown("Upload a CSV file and query insights; automatic Python code + charts are generated.")

    upl_csv_viz = st.file_uploader("📁 Upload CSV for visualization", type="csv", key="viz_uploader")

    if upl_csv_viz is not None:
        df_viz = pd.read_csv(upl_csv_viz)
        st.write("Preview (first 5 rows):")
        st.dataframe(df_viz.head())

        user_q_viz = st.text_area(
            "🔎 Ask a question about your data:",
            "Compare the average cost for two people across categories.",
        )

        if st.button("🚀 Analyze", key="viz_analyze_btn"):
            # Basic validation
            if not (tg_key and e2b_key):
                st.error("Enter Together AI and E2B keys in sidebar.")
            else:
                with Sandbox(api_key=e2b_key) as sand:
                    dataset_path = _upload_to_sandbox(sand, upl_csv_viz)
                    results, full_response = _chat_with_llm(
                        sand,
                        user_q_viz,
                        dataset_path,
                        tg_api_key=tg_key,
                        model_name=chosen_model_id,
                    )

                # ---- Display LLM explanation ----
                st.markdown("### 🤖 LLM Response")
                st.markdown(full_response)

                # ---- Display executed code outputs ----
                if results:
                    st.markdown("### 📈 Code Results")
                    for item in results:
                        # • PNG (E2B returns AttrDict with base64 png) —
                        if hasattr(item, "png") and item.png:
                            img = Image.open(BytesIO(base64.b64decode(item.png)))
                            st.image(img, use_container_width=False)

                        # • Matplotlib figure
                        elif hasattr(item, "figure"):
                            st.pyplot(item.figure)

                        # • Plotly figure
                        elif hasattr(item, "show"):
                            st.plotly_chart(item)

                        # • Pandas output
                        elif isinstance(item, (pd.DataFrame, pd.Series)):
                            st.dataframe(item)

                        else:
                            st.write(item)

# ────────────────────────────────
# Tab 2 – Data Analyst Agent
# ────────────────────────────────
with analyst_tab:
    st.subheader("Data Analyst Agent (phi + DuckDB)")
    st.markdown("Upload CSV/XLSX and query in plain English; agent produces SQL then answers.")

    upl_file_analyst = st.file_uploader("📁 Upload CSV or Excel for analysis", type=["csv", "xlsx"], key="analyst_uploader")

    if upl_file_analyst is not None:
        tmp_path, cols, df_analyst = _preprocess_and_save(upl_file_analyst)

        if tmp_path and df_analyst is not None:
            st.markdown("#### Preview")
            st.dataframe(df_analyst, use_container_width=True)
            st.caption(f"Columns: {', '.join(cols)}")

            # Build semantic model for φ agent
            semantic_model = json.dumps(
                {
                    "tables": [{
                        "name": "uploaded_data",
                        "description": "Dataset uploaded by the user.",
                        "path": tmp_path,
                    }]
                },
                indent=4,
            )

            # Instantiate open-source φ DuckDbAgent
            if not openai_key:
                st.error("Provide your OpenAI key (sidebar) to continue.")
            else:
                llm_phi = OpenAIChat(id="gpt-4o", api_key=openai_key)
                phi_agent = DuckDbAgent(
                    model=llm_phi,
                    semantic_model=semantic_model,
                    tools=[PandasTools()],
                    markdown=True,
                    add_history_to_messages=False,
                    followups=False,
                    read_tool_call_history=False,
                    system_prompt=(
                        "You are an expert data analyst. "
                        "Answer the user's question by:\n"
                        "• Generating **exactly one** SQL query inside ```sql ... ```\n"
                        "• Executing it\n"
                        "• Summarising the result plainly."
                    ),
                )

                analyst_query = st.text_area("🔎 Ask your question:", key="analyst_query_box")

                if st.button("🚀 Run query", key="analyst_run_btn"):
                    if analyst_query.strip() == "":
                        st.warning("Please type a question first.")
                    else:
                        with st.spinner("🧠 φ-Agent is thinking… (full trace in terminal)"):
                            run = phi_agent.run(analyst_query)

                        st.markdown("### 📜 Agent Answer")
                        st.markdown(run.content if hasattr(run, "content") else str(run))

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # The Streamlit runtime calls main automatically, nothing to do here.
    pass
