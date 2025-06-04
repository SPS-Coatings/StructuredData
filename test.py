"""
📊 Unified Data Agent ― ONE chat, TWO super-powers
───────────────────────────────────────────────────────────────────────────────
1. 🔍  Analyse data with SQL (φ / DuckDB / OpenAI)
2. 🎨  Visualise data with auto-written Python (Together AI + E2B sandbox)

Upload **multiple** CSV/XLSX files — each becomes its own DuckDB table and
its own path inside the sandbox.

Run:
    streamlit run unified_data_agent.py
───────────────────────────────────────────────────────────────────────────────
"""

# ── Standard library
import os
import re
import io
import csv
import sys
import json
import tempfile
import contextlib
import warnings
from typing import List, Tuple, Optional, Any

# ── Third-party
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
import base64

# E2B + Together (visual agent)
from e2b_code_interpreter import Sandbox
from together import Together

# φ / DuckDB (analyst agent)
from phi.agent.duckdb import DuckDbAgent
from phi.model.openai import OpenAIChat
from phi.tools.pandas import PandasTools

# ── Config
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
st.set_page_config(page_title="📊 Unified Data Agent", page_icon="📊")

# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap (prevents AttributeError on rerun)
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "datasets":           [],
    "data_prepared":      False,
    "files_hash":         None,
    "chat_history":       [],
    "phi_agent":          None,
    "tg_key":             "",
    "e2b_key":            "",
    "openai_key":         "",
    "tg_model":           "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Globals
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
VISUAL_KEYWORDS = {
    "plot","chart","graph","visual","visualise","visualize","scatter","bar",
    "line","hist","histogram","heatmap","pie","boxplot","area","map",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility ─ router
# ─────────────────────────────────────────────────────────────────────────────
def router_agent(user_query: str) -> str:
    return "visual" if any(w in user_query.lower() for w in VISUAL_KEYWORDS) else "analyst"

# ─────────────────────────────────────────────────────────────────────────────
# Utility ─ data preparation
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Sanitize uploaded CSV/XLSX → coerce types → persist to quoted temp CSV.
    Returns (temp_file_path, column_names, dataframe) or (None,None,None) on failure.
    """
    try:
        # — read
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA","N/A","missing"])
        elif file.name.lower().endswith((".xls",".xlsx")):
            df = pd.read_excel(file, na_values=["NA","N/A","missing"])
        else:
            st.error("Unsupported file type. Upload CSV or Excel.")
            return None, None, None

        # — escape quotes for CSV safety
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # — basic type coercion
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # — persist to temp CSV (DuckDB likes real files)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df

    except Exception as exc:
        st.error(f"Pre-processing failed: {exc}")
        return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# Utility ─ visual agent helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_python(llm_response: str) -> str:
    """Return the first ```python …``` block or ''."""
    m = PYTHON_BLOCK.search(llm_response)
    return m.group(1) if m else ""

def execute_in_sandbox(sb: Sandbox, code: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Run Python in sandbox.
    Returns (results, error_message). error_message is None on success.
    """
    with st.spinner("🔧 Executing Python in sandbox…"):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run = sb.run_code(code)
        if out.getvalue():   print("[E2B-STDOUT]\n", out.getvalue())
        if err.getvalue():   print("[E2B-STDERR]\n", err.getvalue(), file=sys.stderr)
    return (None, run.error) if run.error else (run.results, None)

def upload_to_sandbox(sb: Sandbox, file) -> str:
    """
    Write the uploaded file’s **bytes** into the sandbox FS and return its path.
    Using bytes (not the file-object) fixes FileNotFoundError at runtime.
    """
    sandbox_path = f"./{file.name}"
    sb.files.write(sandbox_path, file.getvalue())
    file.seek(0)  # restore pointer for possible re-reads
    return sandbox_path

# ─────────────────────────────────────────────────────────────────────────────
# Visual agent (Together → E2B → auto-retry on error)
# ─────────────────────────────────────────────────────────────────────────────
def visual_agent(
    query: str,
    dataset_infos: List[dict],
    tg_key: str,
    tg_model: str,
    e2b_key: str,
    max_retries: int = 2,
) -> Tuple[str, Optional[List[Any]]]:
    """
    1. LLM writes Python; 2. sandbox runs it.
    On sandbox error we pass the error back to the LLM (up to `max_retries`).
    Returns (cumulative_llm_text, results_or_None).
    """

    def llm_call(msgs: List[dict]) -> str:
        client = Together(api_key=tg_key)
        resp = client.chat.completions.create(model=tg_model, messages=msgs)
        return resp.choices[0].message.content

    with Sandbox(api_key=e2b_key) as sb:
        # ① Upload datasets & build mapping
        for info in dataset_infos:
            info["sb_path"] = upload_to_sandbox(sb, info["file"])

        mapping_lines = [f"- **{d['name']}** ➡ `{d['sb_path']}`" for d in dataset_infos]

        system_prompt = (
            "You are a senior Python data-scientist and visualisation expert.\n"
            "Datasets available inside the sandbox:\n"
            + "\n".join(mapping_lines) +
            "\n• Think step-by-step.\n"
            "• Return exactly **one** ```python ...``` block that **uses the paths above verbatim**.\n"
            "• After the code block, add a short plain-English explanation."
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]
        cumulative_llm_text = ""

        # ② loop (initial + retries)
        for attempt in range(max_retries + 1):
            with st.spinner("🤖 LLM (Together) is thinking…"):
                llm_text = llm_call(msgs)
            cumulative_llm_text += ("\n\n" + llm_text if cumulative_llm_text else llm_text)

            code = extract_python(llm_text)
            if not code:
                st.warning("No ```python``` block found in LLM response.")
                return cumulative_llm_text, None

            results, err = execute_in_sandbox(sb, code)
            if err is None:
                return cumulative_llm_text, results

            # — error handling
            st.warning(f"Sandbox error:\n\n```\n{err}\n```")
            if attempt == max_retries:
                cumulative_llm_text += (
                    "\n\n**Sandbox execution still failed after retry. "
                    "Displayed error above.**"
                )
                return cumulative_llm_text, None

            # Feed error back to the LLM
            msgs.extend([
                {"role": "assistant", "content": llm_text},
                {
                    "role": "user",
                    "content": (
                        "Your previous code raised this error inside a secure sandbox:\n\n"
                        f"```\n{err}\n```\n\n"
                        "Please fix it and return **only** a new ```python``` block."
                    ),
                },
            ])

    # fallback (should not reach)
    return cumulative_llm_text, None

# ─────────────────────────────────────────────────────────────────────────────
# Analyst agent helper
# ─────────────────────────────────────────────────────────────────────────────
def build_phi_agent(csv_paths, openai_key: str) -> DuckDbAgent:
    """Create φ DuckDB agent for one or many CSVs."""
    tables_meta = (
        [
            {"name": i["name"], "description": f"Dataset {os.path.basename(i['path'])}.", "path": i["path"]}
            for i in csv_paths
        ] if isinstance(csv_paths, list) else
        [{
            "name":"uploaded_data","description":"Single uploaded dataset.","path":csv_paths,
        }]
    )

    llm = OpenAIChat(id="gpt-4o", api_key=openai_key)
    return DuckDbAgent(
        model           = llm,
        semantic_model  = json.dumps({"tables": tables_meta}, indent=4),
        tools           = [PandasTools()],
        markdown        = True,
        followups       = False,
        system_prompt   = (
            "You are an expert data analyst.\n"
            "Answer by:\n"
            "1. Writing **one** SQL query in ```sql```.\n"
            "2. Executing it.\n"
            "3. Presenting the result plainly."
        ),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar ─ API keys & model
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    tg_key     = st.text_input("Together AI key (visualisation)", type="password")
    e2b_key    = st.text_input("E2B key (sandbox)",               type="password")
    openai_key = st.text_input("OpenAI key (analysis + φ)",       type="password")

    st.markdown("### 🎛️ Visual-agent model")
    TG_MODELS = {
        "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
        "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }
    model_human = st.selectbox("Together model", list(TG_MODELS.keys()), index=0)
    tg_model    = TG_MODELS[model_human]

st.session_state.update({
    "tg_key": tg_key, "e2b_key": e2b_key,
    "openai_key": openai_key, "tg_model": tg_model,
})

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Unified Data Agent")

uploaded_files = st.file_uploader(
    "📁 Upload one or more CSV / Excel files",
    type=["csv","xlsx"], accept_multiple_files=True,
)

# — handle uploads & preprocessing
if uploaded_files and st.session_state.files_hash != hash(tuple(f.name for f in uploaded_files)):
    infos: List[dict] = []
    for f in uploaded_files:
        path, cols, df = preprocess_and_save(f)
        if path:
            infos.append({
                "file": f, "path": path,
                "name": re.sub(r"\W|^(?=\d)", "_", os.path.splitext(f.name)[0]),
                "columns": cols, "df": df,
            })
    if infos:
        st.session_state.update({
            "datasets": infos, "data_prepared": True,
            "files_hash": hash(tuple(f.name for f in uploaded_files)),
            "phi_agent": build_phi_agent(infos, openai_key) if openai_key else None,
        })

# — preview datasets
if st.session_state.data_prepared:
    st.subheader("Dataset previews (first 8 rows)")
    for d in st.session_state.datasets:
        with st.expander(f"📄 {d['file'].name}  (table {d['name']})"):
            st.dataframe(d["df"].head(8), use_container_width=True)
            st.caption(f"Columns: {', '.join(d['columns'])}")

# ─────────────────────────────────────────────────────────────────────────────
# Chat interface
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.data_prepared:

    # render history
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for r in m.get("vis_results", []):
                match r["type"]:
                    case "image":      st.image(r["data"])
                    case "matplotlib": st.pyplot(r["data"])
                    case "plotly":     st.plotly_chart(r["data"])
                    case "table":      st.dataframe(r["data"])
                    case _:            st.write(r["data"])

    # input
    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        agent_type = router_agent(prompt)
        if agent_type == "visual":
            if not (tg_key and e2b_key):
                st.error("Provide Together AI and E2B keys."); st.stop()
        else:
            if not openai_key:
                st.error("Provide OpenAI key."); st.stop()

        # ─ visual branch
        if agent_type == "visual":
            text, results = visual_agent(
                query=prompt,
                dataset_infos=st.session_state.datasets,
                tg_key=tg_key, tg_model=tg_model, e2b_key=e2b_key,
            )
            msg = {"role":"assistant","content":text,"vis_results":[]}
            with st.chat_message("assistant"):
                st.markdown(text)
                if results:
                    for obj in results:
                        if getattr(obj,"png",None):
                            img = Image.open(BytesIO(base64.b64decode(obj.png)))
                            st.image(img); msg["vis_results"].append({"type":"image","data":img})
                        elif getattr(obj,"figure",None):
                            st.pyplot(obj.figure); msg["vis_results"].append({"type":"matplotlib","data":obj.figure})
                        elif getattr(obj,"show",None):
                            st.plotly_chart(obj);   msg["vis_results"].append({"type":"plotly","data":obj})
                        elif isinstance(obj,(pd.DataFrame,pd.Series)):
                            st.dataframe(obj);      msg["vis_results"].append({"type":"table","data":obj})
                        else:
                            st.write(obj);          msg["vis_results"].append({"type":"text","data":str(obj)})
            st.session_state.chat_history.append(msg)

        # ─ analyst branch
        else:
            if st.session_state.phi_agent is None:
                st.session_state.phi_agent = build_phi_agent(st.session_state.datasets, openai_key)
            with st.spinner("🧠 φ-agent is thinking…"):
                run = st.session_state.phi_agent.run(prompt)
            answer = run.content if hasattr(run,"content") else str(run)
            with st.chat_message("assistant"): st.markdown(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})

else:
    st.info("⬆️  Upload one or more datasets to start chatting!")
