"""
Data Analysis Agent
--------------------
Streamlit app: user uploads a CSV/Excel file, types a natural-language query,
and the app returns a formatted text answer and/or a chart.

Pipeline:
    1. File ingestion & profiling (schema, sample rows, dtypes)
    2. Intent classification (Gemini call #1) -> structured JSON
    3. Code generation (Gemini call #2) -> pandas/plotly snippet
    4. Sandboxed execution (+ self-correction retry loop on exceptions)
    5. Review (Gemini call #3) -> pass/fail + suggested fix
           - on fail: regenerate code with the fix as extra context
    6. Formatting & display in Streamlit

Run with:
    streamlit run data_analysis_agent.py

Requires:
    pip install streamlit pandas openpyxl plotly google-generativeai
"""

import io
import json
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai
from plotly.subplots import make_subplots  


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_EXEC_RETRIES = 2   # retries after an execution exception
MAX_REVIEW_RETRIES = 2  # retries after the reviewer says "fail"

st.set_page_config(page_title="Data Analysis Agent", layout="wide")


# --------------------------------------------------------------------------
# Gemini helpers
# --------------------------------------------------------------------------

def get_model():
    """Return a configured Gemini model using the API key from the sidebar."""
    api_key = st.session_state.get("api_key")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def call_gemini_json(model, prompt: str) -> dict:
    """Call Gemini and parse a JSON object out of the response text.

    Strips markdown code fences if the model wraps the JSON in them.
    """
    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    text = text.strip()
    return json.loads(text)


def call_gemini_text(model, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text.strip()


# --------------------------------------------------------------------------
# Step 1: File ingestion & profiling
# --------------------------------------------------------------------------

def load_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    # Excel: handle multi-sheet files by letting the user choose
    xls = pd.ExcelFile(uploaded_file)
    if len(xls.sheet_names) > 1:
        sheet = st.selectbox("Multiple sheets found — pick one", xls.sheet_names)
    else:
        sheet = xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


def profile_dataframe(df: pd.DataFrame) -> str:
    """Build a compact text description of the dataframe's schema for prompts.
    Only a sample of rows is ever sent to the LLM, never the full dataset.
    """
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write("Columns (name: dtype, non-null count):\n")
    for col in df.columns:
        buf.write(f"  - {col}: {df[col].dtype}, {df[col].notna().sum()} non-null\n")
    buf.write("\nSample rows:\n")
    buf.write(df.head(5).to_string())
    return buf.getvalue()


# --------------------------------------------------------------------------
# Step 2: Intent classification
# --------------------------------------------------------------------------

def classify_intent(model, query: str, schema_text: str) -> dict:
    prompt = f"""You are a data-analysis intent classifier.

Dataset schema:
{schema_text}

User query: "{query}"

Classify this query. Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "intent": "aggregation | filter | groupby | sort | correlation | trend | comparison | distribution | other",
  "output_type": "text | chart | both",
  "chart_type": "bar | line | scatter | pie | histogram | box | none",
  "columns_involved": ["list", "of", "column", "names", "from", "the", "schema"],
  "operation": "short description of the computation needed"
}}"""
    return call_gemini_json(model, prompt)


# --------------------------------------------------------------------------
# Step 3: Code generation
# --------------------------------------------------------------------------

def generate_code(model, query: str, schema_text: str, intent: dict,
                   extra_context: str = "") -> str:
    prompt = f"""You write short pandas/plotly Python snippets that operate on an
existing dataframe called `df`. Do not read files, do not import anything beyond
pandas (as pd) and plotly.graph_objects (as go) and plotly.express (as px), and make_subplots 
which are already imported. Do not use exec/eval/os/sys/open.

Dataset schema:
{schema_text}

User query: "{query}"

Classified intent:
{json.dumps(intent, indent=2)}

{extra_context}

Rules:
- If output_type is "text", assign the final answer to a variable named `result`
  (a string, number, or small dataframe).
- If output_type is "chart", assign a plotly figure to a variable named `fig`.
- If output_type is "both", assign both `result` and `fig`.
- Keep the snippet short and use only the given dataframe `df`.

Respond with ONLY the Python code, no markdown fences, no explanation."""
    code = call_gemini_text(model, prompt)
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[len("python"):]
        code = code.rsplit("```", 1)[0]
    return code.strip()


# --------------------------------------------------------------------------
# Step 4: Sandboxed execution
# --------------------------------------------------------------------------

def run_code_sandboxed(code: str, df: pd.DataFrame):
    """Execute generated code with a restricted namespace.
    Returns (result, fig, error_traceback_or_None).
    """
    safe_globals = {
        "__builtins__": {
            "len": len, "range": range, "sum": sum, "min": min, "max": max,
            "round": round, "sorted": sorted, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "abs": abs, "enumerate": enumerate,
        },
        "pd": pd,
        "go": go,
        "make_subplots": make_subplots,
    }
    try:
        import plotly.express as px
        safe_globals["px"] = px
    except ImportError:
        pass

    safe_locals = {"df": df.copy()}
    
    # Strip out 'import' and 'from' statements to prevent the __import__ error
    cleaned_code = "\n".join(
        line for line in code.split("\n") 
        if not line.strip().startswith(("import ", "from "))
    )

    try:
        exec(cleaned_code, safe_globals, safe_locals)
        return safe_locals.get("result"), safe_locals.get("fig"), None
    except Exception:
        return None, None, traceback.format_exc()
        
# --------------------------------------------------------------------------
# Step 5: Review
# --------------------------------------------------------------------------

def describe_output_for_review(result, fig) -> str:
    parts = []
    if result is not None:
        parts.append(f"Text/table result: {str(result)[:1000]}")
    if fig is not None:
        n_traces = len(fig.data)
        trace_types = [t.type for t in fig.data]
        parts.append(f"Chart: {n_traces} trace(s), type(s) = {trace_types}")
    return "\n".join(parts) if parts else "No output was produced."


def review_output(model, query: str, intent: dict, code: str, output_desc: str) -> dict:
    prompt = f"""You are reviewing the output of a data analysis pipeline before it is
shown to the user.

Original user query: "{query}"

Classified intent:
{json.dumps(intent, indent=2)}

Generated code:
{code}

Actual output produced:
{output_desc}

Check three things:
1. Correctness of interpretation - does the output actually answer the query?
2. Sanity of the result - is it plausible (no impossible values, empty results, etc.)?
3. Chart appropriateness - if a chart, is the chart type sensible for the data and query?

Respond with ONLY a JSON object, no markdown fences:
{{
  "verdict": "pass" or "fail",
  "issue": "short description of the problem, empty string if pass",
  "suggested_fix": "concrete instruction for how to fix the code, empty string if pass"
}}"""
    return call_gemini_json(model, prompt)


# --------------------------------------------------------------------------
# Step 6: Formatting
# --------------------------------------------------------------------------

def format_text_answer(model, query: str, result) -> str:
    if result is None:
        return ""
    prompt = f"""The user asked: "{query}"

The computed result is:
{result}

Phrase this as a single, clear, natural-language sentence or short paragraph
answering the user's question. Do not invent numbers beyond what is given."""
    return call_gemini_text(model, prompt)


# --------------------------------------------------------------------------
# Main pipeline orchestration
# --------------------------------------------------------------------------

def run_pipeline(model, df: pd.DataFrame, query: str):
    schema_text = profile_dataframe(df)

    with st.status("Classifying your query...", expanded=False) as status:
        intent = classify_intent(model, query, schema_text)
        status.update(label=f"Intent: {intent.get('intent')} ({intent.get('output_type')})")

    extra_context = ""
    for review_attempt in range(MAX_REVIEW_RETRIES + 1):
        code = None
        result, fig, error = None, None, None

        for exec_attempt in range(MAX_EXEC_RETRIES + 1):
            with st.status(f"Generating code (attempt {exec_attempt + 1})...", expanded=False):
                code = generate_code(model, query, schema_text, intent, extra_context)

            with st.status("Running analysis...", expanded=False):
                result, fig, error = run_code_sandboxed(code, df)

            if error is None:
                break
            # feed the error back for self-correction
            extra_context = (
                f"The previous attempt raised this error, fix it:\n{error}\n"
                f"Previous code:\n{code}"
            )

        if error is not None:
            st.error("The analysis code kept failing. Try rephrasing your question.")
            with st.expander("Show last error"):
                st.code(error)
            return

        with st.status("Reviewing output...", expanded=False) as status:
            output_desc = describe_output_for_review(result, fig)
            review = review_output(model, query, intent, code, output_desc)
            status.update(label=f"Review: {review.get('verdict')}")

        if review.get("verdict") == "pass":
            break

        extra_context = (
            f"A reviewer flagged this issue with the previous attempt: "
            f"{review.get('issue')}\nSuggested fix: {review.get('suggested_fix')}\n"
            f"Previous code:\n{code}"
        )
    else:
        st.warning("Review kept failing — showing the last attempt anyway, please verify.")

    # ---- Display ----
    if result is not None:
        answer_text = format_text_answer(model, query, result)
        st.markdown(f"**Answer:** {answer_text}")
        if isinstance(result, (pd.DataFrame, pd.Series)):
            st.dataframe(result)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show generated code"):
        st.code(code, language="python")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main():
    st.title("📊 Data Analysis Agent")
    st.caption("Upload a CSV/Excel file, ask a question, get an answer or a chart.")

    with st.sidebar:
        st.session_state["api_key"] = st.text_input(
            "Gemini API key", type="password", value=st.session_state.get("api_key", "")
        )

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("Upload a file to get started.")
        return

    df = load_dataframe(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns.")
    with st.expander("Preview data"):
        st.dataframe(df.head(20))

    query = st.text_input("Ask a question about your data")

    if st.button("Analyze") and query:
        model = get_model()
        if model is None:
            st.error("Enter your Gemini API key in the sidebar first.")
            return
        run_pipeline(model, df, query)


if __name__ == "__main__":
    main()
