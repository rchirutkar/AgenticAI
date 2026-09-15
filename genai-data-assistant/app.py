import streamlit as st
import pandas as pd
import plotly.express as px
import traceback
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment keys from .env file
load_dotenv()


# Initialize the modern Gemini Client (automatically pulls GEMINI_API_KEY from environment)
client = genai.Client()

st.set_page_config(page_title="GenAI Data Assistant", layout="wide")
st.title("📊 Advanced GenAI Data Assistant")
st.caption("Upload your data, ask a natural language question, and let the pipeline do the rest.")

# Ensure session state is initialized
if "df" not in st.session_state:
    st.session_state.df = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None

# =====================================================================
# STEP 1: Upload & Profile
# =====================================================================
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and st.session_state.df is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        # Profile only structural metadata (Never pass the full dataset to the LLM)
        df_sample = st.session_state.df
        st.session_state.metadata = {
            "columns": list(df_sample.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
            "sample_rows": df_sample.head(3).to_dict(orient="records")
        }
        st.sidebar.success(True, icon="✅")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Display active dataset info if loaded
if st.session_state.df is not None:
    st.sidebar.write(f"**Total Rows:** {len(st.session_state.df)}")
    st.sidebar.write(f"**Total Columns:** {len(st.session_state.df.columns)}")
    with st.expander("🔎 Preview Dataset Schema"):
        st.write(st.session_state.metadata)

# =====================================================================
# PIPELINE FUNCTIONS (Steps 3, 4, 5, 6)
# =====================================================================

def classify_intent(query, metadata):
    """STEP 3: Intent Classification using Gemini JSON Structured Outputs"""
    prompt = f"""
    Analyze the user query and the dataset metadata schema provided.
    Determine the analysis intent, output preference, and relevant target columns.
    
    User Query: '{query}'
    Dataset Metadata: {json.dumps(metadata)}
    """
    
    # Define response schema matching your specification
    class IntentResponse(BaseModel):
        intent_type: str  # e.g., aggregation, filtering, trend, distribution, description
        output_type: str  # text, chart, or both
        chart_type: str   # bar, line, scatter, box, pie, or none
        relevant_columns: list[str]

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentResponse,
            system_instruction="You are a strict data router. Extract intended variables and output types into structural JSON."
        ),
    )
    return json.loads(response.text)


def generate_code(query, metadata, intent_data, correction_hint=None):
    """STEP 4: Generate short pandas/plotly snippets"""
    correction_context = f"\nCorrection Context from previous failed run: {correction_hint}" if correction_hint else ""
    
    prompt = f"""
    You are an expert Python data scientist. Generate a code snippet based on the following constraints:
    1. The dataset is already fully loaded in memory as a pandas DataFrame variable named `df`. Do NOT redefine `df`.
    2. Save any textual/numeric insights into a local variable named `result`.
    3. Save any generated interactive plot using Plotly Express into a variable named `fig`.
    4. Do not import pandas. You can assume `import plotly.express as px` is already available.
    5. Write only clean, operational python code block without code fences or labels.
    
    User Query: '{query}'
    Intent Metadata: {json.dumps(intent_data)}
    Dataset Metadata: {json.dumps(metadata)}{correction_context}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Output raw, bare Python code only. Do not wrap code blocks in markdown blocks like ```python."
        )
    )
    # Basic sanitize to strip markdown blocks if model leaks them
    clean_code = response.text.replace("```python", "").replace("```", "").strip()
    return clean_code


def run_sandboxed(code_string, df):
    """STEP 5: Restricted execution namespace monitoring errors"""
    safe_globals = {
        "__builtins__": {
            "print": print, "range": range, "len": len, "int": int, "float": float, 
            "str": str, "dict": dict, "list": list, "sum": sum, "max": max, "min": min
        }
    }
    safe_locals = {"df": df, "px": px, "result": None, "fig": None}
    
    try:
        exec(code_string, safe_globals, safe_locals)
        return {
            "success": True,
            "result": safe_locals.get("result"),
            "fig": safe_locals.get("fig"),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "fig": None,
            "error": traceback.format_exc()
        }


def review_output(query, code, execution_res):
    """STEP 6: Critic review for logical consistency and chart safety"""
    prompt = f"""
    Analyze if the generated code successfully answered the user's data question.
    
    User Query: '{query}'
    Executed Python Code: 
    {code}
    
    Execution Output Captured:
    Text Result: {execution_res.get('result')}
    Has Chart Fig Object: {execution_res.get('fig') is not None}
    
    Determine if this output is accurate, sane, and visually maps perfectly to the query.
    """
    
    class ReviewResponse(BaseModel):
        status: str  # "pass" or "fail"
        suggested_fix: str # If fail, provide explicit hints on how to correct the code logic. If pass, leave blank.

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReviewResponse,
            system_instruction="Act as a data analyst quality critic. Fail any code that is logically flawed or produces empty results."
        ),
    )
    return json.loads(response.text)


def format_final_text(query, raw_result):
    """STEP 7: Natural language translation helper"""
    if raw_result is None:
        return "I processed the data execution loop successfully."
    
    prompt = f"Convert this raw programming data outcome: '{raw_result}' into a conversational, human sentence answering this user inquiry: '{query}'."
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return response.text

# =====================================================================
# STEP 2 & 7: UI Execution Engine Loop
# =====================================================================
if st.session_state.df is not None:
    user_query = st.text_input("💬 Step 2: What would you like to analyze or visualize?")
    
    if user_query:
        with st.spinner("Executing pipeline routines..."):
            
            # 1. Classify Intent
            intent_data = classify_intent(user_query, st.session_state.metadata)
            
            # Initialize Self-Healing Iterative State Variables
            MAX_RETRIES = 3
            current_try = 0
            correction_hint = None
            pipeline_passed = False
            
            final_execution = None
            final_code = ""
            
            # Self-healing loop controlling execution (Step 5) & Critique (Step 6)
            while current_try < MAX_RETRIES and not pipeline_passed:
                current_try += 1
                
                # Generate Code
                final_code = generate_code(user_query, st.session_state.metadata, intent_data, correction_hint)
                
                # Run Sandbox
                exec_outcome = run_sandboxed(final_code, st.session_state.df)
                
                if not exec_outcome["success"]:
                    # Code crashed -> loop straight back to code generator using traceback error as hint
                    correction_hint = f"Python Runtime Error:\n{exec_outcome['error']}"
                    continue
                
                # Run Critic Review if runtime succeeded
                critic_data = review_output(user_query, final_code, exec_outcome)
                
                if critic_data["status"] == "pass":
                    pipeline_passed = True
                    final_execution = exec_outcome
                else:
                    # Critic failed the output logic -> Loop back with recommended fixes
                    correction_hint = f"Critic Logical Flaw: {critic_data['suggested_fix']}"
           
            # =====================================================================
            # STEP 7: Format & Display
            # =====================================================================
            if pipeline_passed and final_execution:
                st.success(f"Pipeline executed successfully in {current_try} iteration(s)!")
                
                # 1. Render Natural Language Sentence
                formatted_sentence = format_final_text(user_query, final_execution["result"])
                st.write("### 💡 Answer")
                st.write(formatted_sentence)
                
                # 2. Render Interactive Chart (if generated)
                if final_execution["fig"] is not None:
                    st.write("### 📊 Visualization")
                    st.plotly_chart(final_execution["fig"], use_container_width=True)
                
                # 3. Render Code Transparency Expander
                with st.expander("🛠️ View Executed Code"):
                    st.markdown(f"**Total Iterations Needed:** `{current_try}`")
                    st.markdown("**Classified Intent JSON:**")
                    st.json(intent_data)
                    st.markdown("**Executed Python Code:**")
                    st.code(final_code, language="python")
            else:
                st.error("The agent pipeline hit maximum retry boundaries without generating stable, logical outcomes.")
                if correction_hint:
                    st.warning(f"Last recorded error context: {correction_hint}")


            

# ---

# ### 🚀 Phase 3: Execution Steps

# Follow these final commands to spin up your analytical agent dashboard:

# #### 1. Run the App
# Launch the interactive web framework directly from your terminal:
# ```bash
# streamlit run app.py