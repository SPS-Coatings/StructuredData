# # unified_data_agent.py
# """
# 📊 Unified Data Agent ― ONE chat, TWO super-powers
# ───────────────────────────────────────────────────────────────────────────────
# A single Streamlit interface that lets you

# 1. 🔍  *Analyse* data with SQL-style reasoning (φ / DuckDB / OpenAI)
# 2. 🎨  *Visualise* data with auto-written Python (Together AI + E2B sandbox)

# An internal “router-agent” picks the best tool for each user question, so you
# can mix analysis and visualisation seamlessly in the same conversation.

# Run with:
#     streamlit run unified_data_agent.py
# ───────────────────────────────────────────────────────────────────────────────
# """

# # ── Standard library
# import os
# import re
# import io
# import csv
# import sys
# import json
# import tempfile
# import contextlib
# import warnings
# from typing import List, Tuple, Optional, Any

# # ── Third-party
# import pandas as pd
# import streamlit as st
# from PIL import Image
# from io import BytesIO
# import base64

# # E2B + Together (visual agent)
# from e2b_code_interpreter import Sandbox
# from together import Together

# # φ / DuckDB (analyst agent)
# from phi.agent.duckdb import DuckDbAgent
# from phi.model.openai import OpenAIChat
# from phi.tools.pandas import PandasTools

# # ── Config
# warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# st.set_page_config(page_title="📊 Unified Data Agent", page_icon="📊")

# # Regex to extract python blocks from LLM messages
# PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# # ──────────────────────────────────────────────────────────────────────────────
# # Helper ─ Router logic
# # ──────────────────────────────────────────────────────────────────────────────
# VISUAL_KEYWORDS = {
#     "plot", "chart", "graph", "visual", "visualise", "visualize",
#     "scatter", "bar", "line", "hist", "histogram", "heatmap", "pie",
#     "boxplot", "area", "map",
# }


# def router_agent(user_query: str) -> str:
#     """
#     Decide which capability the query needs.

#     Returns:
#         "visual"  – if the question clearly asks for a plot/figure
#         "analyst" – otherwise
#     """
#     lower_q = user_query.lower()
#     if any(word in lower_q for word in VISUAL_KEYWORDS):
#         return "visual"
#     return "analyst"


# # ──────────────────────────────────────────────────────────────────────────────
# # Helper ─ Data preparation
# # ──────────────────────────────────────────────────────────────────────────────
# def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
#     """
#     Sanitize uploaded CSV/XLSX → coerce types → persist to quoted temp CSV.

#     Returns:
#         (temp_file_path, column_names, dataframe)
#     """
#     try:
#         # --- Read
#         if file.name.lower().endswith(".csv"):
#             df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
#         elif file.name.lower().endswith((".xls", ".xlsx")):
#             df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
#         else:
#             st.error("Unsupported file type. Upload CSV or Excel.")
#             return None, None, None

#         # --- Escape quotes for CSV safety
#         for col in df.select_dtypes(include="object"):
#             df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

#         # --- Attempt type coercion
#         for col in df.columns:
#             if "date" in col.lower():
#                 df[col] = pd.to_datetime(df[col], errors="coerce")
#             elif df[col].dtype == "object":
#                 df[col] = pd.to_numeric(df[col], errors="ignore")

#         # --- Persist to temporary quoted CSV (DuckDB likes real files)
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#         df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
#         return tmp.name, df.columns.tolist(), df

#     except Exception as exc:
#         st.error(f"Pre-processing failed: {exc}")
#         return None, None, None


# # ──────────────────────────────────────────────────────────────────────────────
# # Helper ─ Visual agent functions
# # ──────────────────────────────────────────────────────────────────────────────
# def extract_python(llm_response: str) -> str:
#     """Return first ```python ...``` block or empty string."""
#     match = PYTHON_BLOCK.search(llm_response)
#     return match.group(1) if match else ""


# def execute_in_sandbox(sb: Sandbox, code: str) -> Optional[List[Any]]:
#     """Run Python `code` in an E2B sandbox; capture results."""
#     with st.spinner("🔧 Executing Python in sandbox…"):
#         stdout_cap, stderr_cap = io.StringIO(), io.StringIO()

#         with contextlib.redirect_stdout(stdout_cap), contextlib.redirect_stderr(stderr_cap):
#             run = sb.run_code(code)

#         # Log sandbox stdout/stderr to terminal
#         if stdout_cap.getvalue():
#             print("[E2B-STDOUT]\n", stdout_cap.getvalue())
#         if stderr_cap.getvalue():
#             print("[E2B-STDERR]\n", stderr_cap.getvalue(), file=sys.stderr)

#         if run.error:
#             st.error(f"Sandbox error: {run.error}")
#             return None
#         return run.results


# def upload_to_sandbox(sb: Sandbox, file) -> str:
#     """Push uploaded file bytes into sandbox FS; return sandbox path."""
#     sandbox_path = f"./{file.name}"
#     sb.files.write(sandbox_path, file)
#     return sandbox_path


# def visual_agent(
#     query: str,
#     uploaded_file,
#     tg_key: str,
#     tg_model: str,
#     e2b_key: str,
# ) -> Tuple[str, Optional[List[Any]]]:
#     """
#     Use Together AI to write Python, run it in E2B, return (llm_text, results).
#     """
#     with Sandbox(api_key=e2b_key) as sb:
#         dataset_path = upload_to_sandbox(sb, uploaded_file)

#         system_prompt = (
#             "You are a senior Python data-scientist and visualisation expert.\n"
#             f"The CSV dataset is at path `{dataset_path}`.\n"
#             "• Think step-by-step.\n"
#             "• Return a single ```python ...``` block that answers the user's request.\n"
#             "• After the code block, add a short plain-English explanation.\n"
#             "IMPORTANT: Always load the CSV using the *exact* path above."
#         )

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": query},
#         ]

#         with st.spinner("🤖 LLM (Together) is thinking…"):
#             client = Together(api_key=tg_key)
#             resp = client.chat.completions.create(model=tg_model, messages=messages)

#         llm_text = resp.choices[0].message.content
#         python_code = extract_python(llm_text)

#         if not python_code:
#             st.warning("No Python code detected in LLM output.")
#             return llm_text, None

#         results = execute_in_sandbox(sb, python_code)
#         return llm_text, results


# # ──────────────────────────────────────────────────────────────────────────────
# # Helper ─ Analyst agent functions
# # ──────────────────────────────────────────────────────────────────────────────
# def build_phi_agent(csv_path: str, openai_key: str) -> DuckDbAgent:
#     """Create and return a φ DuckDB agent bound to `csv_path`."""
#     semantic_model = json.dumps(
#         {
#             "tables": [{
#                 "name": "uploaded_data",
#                 "description": "Dataset uploaded by the user.",
#                 "path": csv_path,
#             }]
#         },
#         indent=4,
#     )

#     llm = OpenAIChat(id="gpt-4o", api_key=openai_key)
#     agent = DuckDbAgent(
#         model=llm,
#         semantic_model=semantic_model,
#         tools=[PandasTools()],
#         markdown=True,
#         add_history_to_messages=False,
#         followups=False,
#         read_tool_call_history=False,
#         system_prompt=(
#             "You are an expert data analyst.\n"
#             "Answer the user's question by:\n"
#             "1. Writing **exactly one** SQL query inside ```sql ... ```\n"
#             "2. Executing it\n"
#             "3. Presenting the result plainly."
#         ),
#     )
#     return agent


# # ──────────────────────────────────────────────────────────────────────────────
# # Streamlit ― Sidebar (API keys & model choice)
# # ──────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.header("🔑 API Keys")
#     tg_key = st.text_input("Together AI key (visualisation)", type="password")
#     e2b_key = st.text_input("E2B key (sandbox)", type="password")
#     openai_key = st.text_input("OpenAI key (analysis + φ)", type="password")

#     st.markdown("### 🎛️ Visual agent model")
#     TG_MODELS = {
#         "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
#         "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
#         "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
#         "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#     }
#     model_human = st.selectbox("Together model", list(TG_MODELS.keys()), index=0)
#     tg_model = TG_MODELS[model_human]

# # Persist keys/model to session_state
# st.session_state.update({
#     "tg_key": tg_key,
#     "e2b_key": e2b_key,
#     "openai_key": openai_key,
#     "tg_model": tg_model,
# })

# # ──────────────────────────────────────────────────────────────────────────────
# # Main UI
# # ──────────────────────────────────────────────────────────────────────────────
# st.title("📊 Unified Data Agent")

# uploaded_file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])

# # --- Handle upload & preprocessing once --------------------------------------
# if uploaded_file is not None and "data_prepared" not in st.session_state:
#     tmp_path, cols, df_preview = preprocess_and_save(uploaded_file)
#     if tmp_path:
#         st.session_state.data_prepared = True
#         st.session_state.csv_path = tmp_path
#         st.session_state.columns = cols
#         st.session_state.df_preview = df_preview
#         # Build φ agent once per upload
#         if openai_key:
#             st.session_state.phi_agent = build_phi_agent(tmp_path, openai_key)
#         else:
#             st.error("OpenAI key required for analyst agent.")

# # --- Show dataset preview ----------------------------------------------------
# if st.session_state.get("data_prepared"):
#     st.subheader("Dataset preview (first 10 rows)")
#     st.dataframe(st.session_state.df_preview.head(10), use_container_width=True)
#     st.caption(f"Columns: {', '.join(st.session_state.columns)}")

# # ──────────────────────────────────────────────────────────────────────────────
# # Chat interface
# # ──────────────────────────────────────────────────────────────────────────────
# if st.session_state.get("data_prepared"):

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history: List[dict] = []

#     # Display existing messages
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#             # Render any stored visual results
#             if msg.get("vis_results"):
#                 for res in msg["vis_results"]:
#                     if res["type"] == "image":
#                         st.image(res["data"], use_container_width=False)
#                     elif res["type"] == "matplotlib":
#                         st.pyplot(res["data"])
#                     elif res["type"] == "plotly":
#                         st.plotly_chart(res["data"])
#                     elif res["type"] == "table":
#                         st.dataframe(res["data"])
#                     else:
#                         st.write(res["data"])

#     # --- Chat input ----------------------------------------------------------
#     user_query = st.chat_input("Ask a question…")
#     if user_query:
#         # Store user prompt
#         st.session_state.chat_history.append({"role": "user", "content": user_query})
#         with st.chat_message("user"):
#             st.markdown(user_query)

#         # Decide agent
#         chosen_agent = router_agent(user_query)

#         # Validate keys
#         if chosen_agent == "visual":
#             if not (tg_key and e2b_key):
#                 st.error("Provide Together AI and E2B keys.")
#                 st.stop()
#         else:  # analyst
#             if not openai_key:
#                 st.error("Provide OpenAI key.")
#                 st.stop()

#         # --- Run chosen agent -------------------------------------------------
#         if chosen_agent == "visual":
#             llm_text, results = visual_agent(
#                 query=user_query,
#                 uploaded_file=uploaded_file,
#                 tg_key=tg_key,
#                 tg_model=tg_model,
#                 e2b_key=e2b_key,
#             )

#             # Prepare assistant message
#             assistant_msg = {"role": "assistant", "content": llm_text, "vis_results": []}

#             # Display
#             with st.chat_message("assistant"):
#                 st.markdown(llm_text)
#                 if results:
#                     for item in results:
#                         # PNG from E2B
#                         if hasattr(item, "png") and item.png:
#                             img = Image.open(BytesIO(base64.b64decode(item.png)))
#                             st.image(img, use_container_width=False)
#                             assistant_msg["vis_results"].append({"type": "image", "data": img})

#                         # Matplotlib figure
#                         elif hasattr(item, "figure"):
#                             st.pyplot(item.figure)
#                             assistant_msg["vis_results"].append({"type": "matplotlib", "data": item.figure})

#                         # Plotly figure
#                         elif hasattr(item, "show"):
#                             st.plotly_chart(item)
#                             assistant_msg["vis_results"].append({"type": "plotly", "data": item})

#                         # Pandas objects
#                         elif isinstance(item, (pd.DataFrame, pd.Series)):
#                             st.dataframe(item)
#                             assistant_msg["vis_results"].append({"type": "table", "data": item})

#                         else:
#                             st.write(item)
#                             assistant_msg["vis_results"].append({"type": "text", "data": str(item)})

#             # Save assistant message with visuals for history
#             st.session_state.chat_history.append(assistant_msg)

#         else:  # analyst
#             agent: DuckDbAgent = st.session_state.phi_agent
#             with st.spinner("🧠 φ-agent is thinking… (SQL + analysis)"):
#                 run = agent.run(user_query)

#             answer_text = run.content if hasattr(run, "content") else str(run)
#             assistant_msg = {"role": "assistant", "content": answer_text}

#             with st.chat_message("assistant"):
#                 st.markdown(answer_text)

#             st.session_state.chat_history.append(assistant_msg)
# else:
#     st.info("⬆️  Upload a dataset to start chatting!")





# unified_data_agent.py
"""
📊 Unified Data Agent ― ONE chat, TWO super-powers
───────────────────────────────────────────────────────────────────────────────
A single Streamlit interface that lets you

1. 🔍  *Analyse* data with SQL-style reasoning (φ / DuckDB / OpenAI)
2. 🎨  *Visualise* data with auto-written Python (Together AI + E2B sandbox)

🔑 **New**: upload **multiple** CSV/XLSX files and query or plot *across* them.
   – Every file becomes its own DuckDB table (named after the file)  
   – The visual agent receives a dictionary of file → path mappings so it can
     generate Python that loads/joins/comparisons across datasets.

Run with:
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

# Regex to extract python blocks from LLM messages
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

# ──────────────────────────────────────────────────────────────────────────────
# Helper ─ Router logic
# ──────────────────────────────────────────────────────────────────────────────
VISUAL_KEYWORDS = {
    "plot", "chart", "graph", "visual", "visualise", "visualize",
    "scatter", "bar", "line", "hist", "histogram", "heatmap", "pie",
    "boxplot", "area", "map",
}


def router_agent(user_query: str) -> str:
    """
    Decide which capability the query needs.

    Returns:
        "visual"  – if the question clearly asks for a plot/figure
        "analyst" – otherwise
    """
    lower_q = user_query.lower()
    if any(word in lower_q for word in VISUAL_KEYWORDS):
        return "visual"
    return "analyst"


# ──────────────────────────────────────────────────────────────────────────────
# Helper ─ Data preparation
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
    """
    Sanitize uploaded CSV/XLSX → coerce types → persist to quoted temp CSV.

    Returns:
        (temp_file_path, column_names, dataframe)
    """
    try:
        # --- Read
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            st.error("Unsupported file type. Upload CSV or Excel.")
            return None, None, None

        # --- Escape quotes for CSV safety
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # --- Attempt type coercion
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # --- Persist to temporary quoted CSV (DuckDB likes real files)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df

    except Exception as exc:
        st.error(f"Pre-processing failed: {exc}")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Helper ─ Visual agent functions
# ──────────────────────────────────────────────────────────────────────────────
def extract_python(llm_response: str) -> str:
    """Return first ```python ...``` block or empty string."""
    match = PYTHON_BLOCK.search(llm_response)
    return match.group(1) if match else ""


def execute_in_sandbox(sb: Sandbox, code: str) -> Optional[List[Any]]:
    """Run Python `code` in an E2B sandbox; capture results."""
    with st.spinner("🔧 Executing Python in sandbox…"):
        stdout_cap, stderr_cap = io.StringIO(), io.StringIO()

        with contextlib.redirect_stdout(stdout_cap), contextlib.redirect_stderr(stderr_cap):
            run = sb.run_code(code)

        # Log sandbox stdout/stderr to terminal
        if stdout_cap.getvalue():
            print("[E2B-STDOUT]\n", stdout_cap.getvalue())
        if stderr_cap.getvalue():
            print("[E2B-STDERR]\n", stderr_cap.getvalue(), file=sys.stderr)

        if run.error:
            st.error(f"Sandbox error: {run.error}")
            return None
        return run.results


def upload_to_sandbox(sb: Sandbox, file) -> str:
    """Push uploaded file bytes into sandbox FS; return sandbox path."""
    sandbox_path = f"./{file.name}"
    sb.files.write(sandbox_path, file)
    return sandbox_path


def visual_agent(
    query: str,
    dataset_infos: List[dict],
    tg_key: str,
    tg_model: str,
    e2b_key: str,
) -> Tuple[str, Optional[List[Any]]]:
    """
    Use Together AI to write Python, run it in E2B, return (llm_text, results).
    Supports *multiple* datasets.
    """
    with Sandbox(api_key=e2b_key) as sb:
        # Upload every file to sandbox & build mapping
        mapping_lines = []
        for info in dataset_infos:
            path_in_sb = upload_to_sandbox(sb, info["file"])
            info["sb_path"] = path_in_sb          # keep for prompt
            mapping_lines.append(f"- **{info['name']}** ➡ `{path_in_sb}`")

        system_prompt = (
            "You are a senior Python data-scientist and visualisation expert.\n"
            "The following CSV datasets are available:\n"
            + "\n".join(mapping_lines)
            + "\n• Think step-by-step.\n"
              "• Return a single ```python ...``` block that answers the user's request, "
              "loading any dataset(s) via the *exact* paths above.\n"
              "• After the code block, add a short plain-English explanation."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]

        with st.spinner("🤖 LLM (Together) is thinking…"):
            client = Together(api_key=tg_key)
            resp = client.chat.completions.create(model=tg_model, messages=messages)

        llm_text = resp.choices[0].message.content
        python_code = extract_python(llm_text)

        if not python_code:
            st.warning("No Python code detected in LLM output.")
            return llm_text, None

        results = execute_in_sandbox(sb, python_code)
        return llm_text, results


# ──────────────────────────────────────────────────────────────────────────────
# Helper ─ Analyst agent functions
# ──────────────────────────────────────────────────────────────────────────────
def build_phi_agent(csv_paths, openai_key: str) -> DuckDbAgent:
    """
    Create and return a φ DuckDB agent.

    csv_paths can be:
        • str  – a single path  (back-compat)
        • list – list of dicts with keys {'name','path'}
    """
    if isinstance(csv_paths, list):
        tables_meta = [
            {
                "name": item["name"],
                "description": f"Dataset from file {os.path.basename(item['path'])}.",
                "path": item["path"],
            }
            for item in csv_paths
        ]
    else:
        tables_meta = [{
            "name": "uploaded_data",
            "description": "Dataset uploaded by the user.",
            "path": csv_paths,
        }]

    semantic_model = json.dumps({"tables": tables_meta}, indent=4)

    llm = OpenAIChat(id="gpt-4o", api_key=openai_key)
    agent = DuckDbAgent(
        model=llm,
        semantic_model=semantic_model,
        tools=[PandasTools()],
        markdown=True,
        add_history_to_messages=False,
        followups=False,
        read_tool_call_history=False,
        system_prompt=(
            "You are an expert data analyst.\n"
            "Answer the user's question by:\n"
            "1. Writing **exactly one** SQL query inside ```sql ... ```\n"
            "2. Executing it\n"
            "3. Presenting the result plainly."
        ),
    )
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit ― Sidebar (API keys & model choice)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys")
    tg_key = st.text_input("Together AI key (visualisation)", type="password")
    e2b_key = st.text_input("E2B key (sandbox)", type="password")
    openai_key = st.text_input("OpenAI key (analysis + φ)", type="password")

    st.markdown("### 🎛️ Visual agent model")
    TG_MODELS = {
        "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
        "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }
    model_human = st.selectbox("Together model", list(TG_MODELS.keys()), index=0)
    tg_model = TG_MODELS[model_human]

# Persist keys/model to session_state
st.session_state.update({
    "tg_key": tg_key,
    "e2b_key": e2b_key,
    "openai_key": openai_key,
    "tg_model": tg_model,
})

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Unified Data Agent")

uploaded_files = st.file_uploader(
    "📁 Upload one *or more* CSV / Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

# --- Handle uploads & preprocessing -----------------------------------------
if uploaded_files and (
    "files_hash" not in st.session_state
    or st.session_state.files_hash != hash(tuple(f.name for f in uploaded_files))
):
    dataset_infos: List[dict] = []

    for f in uploaded_files:
        tmp_path, cols, df_prev = preprocess_and_save(f)
        if tmp_path:
            table_name = re.sub(r"\W|^(?=\d)", "_", os.path.splitext(f.name)[0])
            dataset_infos.append(
                {
                    "file": f,
                    "path": tmp_path,
                    "name": table_name,
                    "columns": cols,
                    "df": df_prev,
                }
            )

    if dataset_infos:
        st.session_state.datasets = dataset_infos
        st.session_state.data_prepared = True
        st.session_state.files_hash = hash(tuple(f.name for f in uploaded_files))

        # Build φ agent for *all* datasets
        if openai_key:
            st.session_state.phi_agent = build_phi_agent(dataset_infos, openai_key)
        else:
            st.error("OpenAI key required for analyst agent.")

# --- Show dataset previews ---------------------------------------------------
if st.session_state.get("data_prepared"):
    st.subheader("Dataset previews (first 8 rows each)")
    for info in st.session_state.datasets:
        with st.expander(f"📄 {info['file'].name}  (table: {info['name']})", expanded=False):
            st.dataframe(info["df"].head(8), use_container_width=True)
            st.caption(f"Columns: {', '.join(info['columns'])}")

# ──────────────────────────────────────────────────────────────────────────────
# Chat interface
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("data_prepared"):

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[dict] = []

    # Display existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render any stored visual results
            if msg.get("vis_results"):
                for res in msg["vis_results"]:
                    if res["type"] == "image":
                        st.image(res["data"], use_container_width=False)
                    elif res["type"] == "matplotlib":
                        st.pyplot(res["data"])
                    elif res["type"] == "plotly":
                        st.plotly_chart(res["data"])
                    elif res["type"] == "table":
                        st.dataframe(res["data"])
                    else:
                        st.write(res["data"])

    # --- Chat input ----------------------------------------------------------
    user_query = st.chat_input("Ask a question…")
    if user_query:
        # Store user prompt
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Decide agent
        chosen_agent = router_agent(user_query)

        # Validate keys
        if chosen_agent == "visual":
            if not (tg_key and e2b_key):
                st.error("Provide Together AI and E2B keys.")
                st.stop()
        else:  # analyst
            if not openai_key:
                st.error("Provide OpenAI key.")
                st.stop()

        # --- Run chosen agent -------------------------------------------------
        if chosen_agent == "visual":
            llm_text, results = visual_agent(
                query=user_query,
                dataset_infos=st.session_state.datasets,
                tg_key=tg_key,
                tg_model=tg_model,
                e2b_key=e2b_key,
            )

            # Prepare assistant message
            assistant_msg = {"role": "assistant", "content": llm_text, "vis_results": []}

            # Display
            with st.chat_message("assistant"):
                st.markdown(llm_text)
                if results:
                    for item in results:
                        # PNG from E2B
                        if hasattr(item, "png") and item.png:
                            img = Image.open(BytesIO(base64.b64decode(item.png)))
                            st.image(img, use_container_width=False)
                            assistant_msg["vis_results"].append({"type": "image", "data": img})

                        # Matplotlib figure
                        elif hasattr(item, "figure"):
                            st.pyplot(item.figure)
                            assistant_msg["vis_results"].append({"type": "matplotlib", "data": item.figure})

                        # Plotly figure
                        elif hasattr(item, "show"):
                            st.plotly_chart(item)
                            assistant_msg["vis_results"].append({"type": "plotly", "data": item})

                        # Pandas objects
                        elif isinstance(item, (pd.DataFrame, pd.Series)):
                            st.dataframe(item)
                            assistant_msg["vis_results"].append({"type": "table", "data": item})

                        else:
                            st.write(item)
                            assistant_msg["vis_results"].append({"type": "text", "data": str(item)})

            # Save assistant message with visuals for history
            st.session_state.chat_history.append(assistant_msg)

        else:  # analyst
            agent: DuckDbAgent = st.session_state.phi_agent
            with st.spinner("🧠 φ-agent is thinking… (SQL + analysis)"):
                run = agent.run(user_query)

            answer_text = run.content if hasattr(run, "content") else str(run)
            assistant_msg = {"role": "assistant", "content": answer_text}

            with st.chat_message("assistant"):
                st.markdown(answer_text)

            st.session_state.chat_history.append(assistant_msg)
else:
    st.info("⬆️  Upload one or more datasets to start chatting!")

