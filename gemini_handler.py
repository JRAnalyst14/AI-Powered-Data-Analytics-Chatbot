import google.generativeai as genai
import streamlit as st # Import Streamlit to access st.secrets
import pandas as pd
import os # Keep os for potential fallback or other environment variables

# --- Configure Gemini API ---
api_key_configured = False
try:
    # Prioritize Streamlit secrets
    if "gemini" in st.secrets and "api_key" in st.secrets.gemini:
        api_key = st.secrets.gemini.api_key
        if api_key: # Ensure the key from secrets is not empty
            genai.configure(api_key=api_key)
            api_key_configured = True
            print("Gemini API Key configured using Streamlit secrets.") # For server logs
        else:
            print("Warning: Gemini API Key found in Streamlit secrets but it is empty.")
    else:
        # Fallback to environment variable if not in Streamlit secrets (optional, can be removed if only secrets are desired)
        print("Gemini API Key not found in Streamlit secrets. Trying environment variable GOOGLE_API_KEY.")
        env_api_key = os.getenv("GOOGLE_API_KEY")
        if env_api_key:
            genai.configure(api_key=env_api_key)
            api_key_configured = True
            print("Gemini API Key configured using GOOGLE_API_KEY environment variable.")
        else:
            print("Warning: GOOGLE_API_KEY environment variable not found or empty.")

    if not api_key_configured:
        st.warning("⚠️ Gemini API Key not configured. Please add it to `.streamlit/secrets.toml` or set the `GOOGLE_API_KEY` environment variable. Gemini features will be unavailable.")
        print("Critical Warning: Gemini API Key is not configured from any source.")


except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Gemini features may not work.")
    print(f"Error during Gemini API configuration: {e}")
    api_key_configured = False


# Function to generate response from the model with DataFrame context
def get_gemini_response_with_context(query: str, df: pd.DataFrame):
    if not api_key_configured:
        return "🤖 Gemini API is not configured. Please check your API key setup in `.streamlit/secrets.toml`."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        context = "You are a helpful data analytics assistant. The user has uploaded a dataset."
        if df is not None and not df.empty:
            context += f"\nThe dataset has {df.shape[0]} rows and {df.shape[1]} columns."
            context += f"\nColumn names are: {', '.join(df.columns.tolist())}."
            context += "\nHere's a small sample of the data (first 3 rows):\n"
            context += df.head(3).to_markdown(index=False)
            context += "\nAnd here are the data types:\n"
            context += df.dtypes.to_string()
        else:
            context += "\nNo data is currently loaded or the data is empty."

        full_prompt = f"{context}\n\nUser query: \"{query}\"\n\nPlease provide a concise and helpful answer related to the data if possible, or answer the general query."

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            full_prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                # temperature=0.7,
                # max_output_tokens=1024
            )
        )

        if response.parts:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
             return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            block_reason = ""
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = f" Blocked due to: {response.prompt_feedback.block_reason.name}."
            
            finish_reason = ""
            if response.candidates and response.candidates[0].finish_reason:
                 finish_reason_val = response.candidates[0].finish_reason
                 if hasattr(finish_reason_val, 'name'):
                     finish_reason_val = finish_reason_val.name
                 if finish_reason_val not in ["STOP", "MAX_TOKENS"]:
                    finish_reason = f" Generation finished due to: {finish_reason_val}."
            
            return f"🤖 Gemini couldn't provide an answer.{block_reason}{finish_reason} Please try rephrasing your question or check the safety settings if this persists."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_message = f"🤖 An error occurred while contacting Gemini: {str(e)}"
        if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e) or "Unauthenticated" in str(e):
            error_message = "Error: Gemini API key is not valid, missing, or permission was denied. Please check your `.streamlit/secrets.toml`."
        
        # Add a button in the UI to show details, or log them for debugging.
        # For now, returning a detailed error string.
        return f"{error_message}\n<details><summary>Click for technical details</summary>\n\n```\n{error_details}\n```\n</details>"


# # Configure your Gemini API key here or in streamlit secrets
# genai.configure(api_key=st.secrets["gemini"]["api_key"])

# def get_gemini_response(prompt):
#     try:
#         # Initialize Gemini Pro model (text-only)
#         model = genai.GenerativeModel(model_name="gemini-pro")
        
#         # Send prompt
#         response = model.generate_content(prompt)
        
#         return response.text
#     except Exception as e:
#         return f"❌ Gemini API error: {e}"

# import google.generativeai as genai