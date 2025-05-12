import streamlit as st
import pandas as pd
from chatbot_logic import handle_query
from utils import get_plot_function_and_cols
from streamlit_option_menu import option_menu
import base64
from PIL import Image
import io



# --- Page Configuration ---
st.set_page_config(page_title="AI Integrated DataBot", layout="wide")

# --- Theme Toggle ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle = st.toggle("🌙 Toggle Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = toggle

# --- Custom CSS for Aesthetics ---
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body, .main, .block-container {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        .stChatMessage.user {
            background-color: #2a2a2a;
            border-radius: 12px;
            padding: 10px;
        }
        .stChatMessage.assistant {
            background-color: #333333;
            border-left: 4px solid #4a90e2;
            border-radius: 12px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main {background-color: #f5f7fa;}
        .block-container {padding: 2rem 3rem;}
        .stChatMessage.user {background-color: #e8f0fe; border-radius: 12px; padding: 10px;}
        .stChatMessage.assistant {background-color: #ffffff; border-left: 4px solid #4a90e2; border-radius: 12px; padding: 10px;}
        </style>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Chatbot", "Upload Dataset", "View Data"],
        icons=["chat-dots", "upload", "table"],
        menu_icon="cast",
        default_index=0,
    )

    if selected == "Upload Dataset":
        uploaded_file = st.file_uploader("Upload your dataset (CSV, JSON, Excel)", type=["csv", "json", "xlsx"])

        if uploaded_file:
            def try_read_file(file, read_func, encodings=None):
                encodings = encodings or ['utf-8', 'ISO-8859-1', 'latin1']
                for enc in encodings:
                    try:
                        return read_func(file, encoding=enc)
                    except Exception:
                        file.seek(0)  # Reset pointer before retrying
                raise ValueError("Failed to read file with common encodings.")

            try:
                if uploaded_file.name.endswith(".csv"):
                    df = try_read_file(uploaded_file, pd.read_csv)

                elif uploaded_file.name.endswith(".json"):
                    df = try_read_file(uploaded_file, pd.read_json)

                elif uploaded_file.name.endswith(".xlsx"):
                    try:
                        df = pd.read_excel(uploaded_file, engine='openpyxl')
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_excel(uploaded_file, engine='xlrd')

                st.session_state.df = df
                st.success(f"✅ Successfully uploaded: {uploaded_file.name}")

            except Exception as e:
                st.error(f"❌ Failed to upload file: {e}")

    st.markdown("""---  
    **Quick Examples**:
    - Show head  
    - Describe data  
    - Plot histogram of 'age'  
    - Correlation heatmap  
    - Train model to predict 'target'  
    """)

# --- Main Area ---
st.title("📊 AI-Powered Data Analytics Chatbot")
st.caption("Your intelligent assistant for EDA, visualizations, and modeling.")

# --- Display Chat or Data ---
if selected == "Chatbot":
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Please upload a dataset and ask your data-related question."}
        ]


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"])
            elif isinstance(message["content"], dict):
                if message["content"].get("type") == "plotly":
                    st.plotly_chart(message["content"]["fig"], use_container_width=True)
                elif message["content"].get("type") == "static_base64":
                    image_data = base64.b64decode(message["content"]["image"])
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption=message["content"].get("message", ""))
                else:
                    st.warning("🤖 Unknown response type.")
            else:
                st.markdown(message["content"])

    if "df" in st.session_state:
        if prompt := st.chat_input("Ask about your dataset..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = handle_query(prompt, st.session_state.df)
                    if isinstance(result, str):
                        st.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    elif isinstance(result, dict):
                        if result.get("type") == "plotly":
                            st.plotly_chart(result["fig"], use_container_width=True)
                        elif result.get("type") == "static_base64":
                            image_data = base64.b64decode(result["image"])
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption=result.get("message", ""))
                        else:
                            st.warning("🤖 Unexpected dictionary format.")
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    else:
                        st.warning("🤖 Unexpected response format.")
                        st.session_state.messages.append({"role": "assistant", "content": "🤖 Unexpected response format."})
    else:
        st.info("📁 Please upload a dataset in the sidebar to begin.")



elif selected == "View Data":
    if "df" in st.session_state:
        st.subheader("🔍 Data Preview")
        st.dataframe(st.session_state.df.head(100))
        st.markdown(f"**Shape:** {st.session_state.df.shape}")
        st.markdown(f"**Columns:** {list(st.session_state.df.columns)}")
    else:
        st.warning("Please upload a dataset first from the sidebar.")
