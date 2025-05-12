import re
import pandas as pd
from utils import (
    generate_data_summary,
    get_column_stats,
    generate_plot_interactive,
    train_and_evaluate_model,
    get_correlation_matrix
)
from gemini_handler import get_gemini_response_with_context

# Master handler
def handle_query(user_input, df):
    user_input_lower = user_input.lower()

    # --- EDA Commands ---
    if re.search(r"(show|view|display).*(head|first few rows|first \d+ rows)", user_input_lower):
        num_rows = 5
        match = re.search(r"first (\d+) rows", user_input_lower)
        if match:
            num_rows = int(match.group(1))
        return df.head(num_rows) # Returns DataFrame

    if re.search(r"(show|view|display).*(tail|last few rows|last \d+ rows)", user_input_lower):
        num_rows = 5
        match = re.search(r"last (\d+) rows", user_input_lower)
        if match:
            num_rows = int(match.group(1))
        return df.tail(num_rows) # Returns DataFrame    

    if re.search(r"(summary|describe|basic info|overview)", user_input_lower):
        return generate_data_summary(df) # Returns string

    if re.search(r"(column names|show columns|list columns)", user_input_lower):
        return f"**Column Names:**\n\n```\n{', '.join(df.columns)}\n```"

    if re.search(r"(data types|dtypes|schema)", user_input_lower):
        dtypes_str = df.dtypes.to_string()
        return f"**Data Types:**\n\n```\n{dtypes_str}\n```"

    if re.search(r"(shape|dimensions|size of data)", user_input_lower):
        return f"**Dataset Shape:**\n\nRows: {df.shape[0]}, Columns: {df.shape[1]}"

    if re.search(r"(missing values|null values|na values|check missing)", user_input_lower):
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if missing_summary.empty:
            return "✅ No missing values found in the dataset."
        return f"**Missing Values per Column:**\n\n```\n{missing_summary.to_string()}\n```"

    if re.search(r"(correlation|correlation matrix|corr matrix)", user_input_lower):
        # This will return a Plotly figure dictionary
        return get_correlation_matrix(df)


    # --- Stats Commands ---
    if re.search(r"(mean|average|median|mode|std|min|max|stats for).*column", user_input_lower) or \
       re.search(r"stats for ('.*'|\".*\"|[a-zA-Z0-9_]+)", user_input_lower): # handles "stats for 'column_name'"
        return get_column_stats(user_input_lower, df) # Returns string

    # --- Plot Commands ---
    if re.search(r"(plot|graph|visualize|chart)", user_input_lower):
        # generate_plot_interactive will return a dict for plotly or a string message
        plot_response = generate_plot_interactive(user_input_lower, df)
        return plot_response

    # --- ML Commands ---
    if re.search(r"(train|build|fit|create|develop).*(model|predict|regression|classification)", user_input_lower):
        return train_and_evaluate_model(user_input_lower, df) # Returns string

    # --- Fallback to Gemini ---
    try:
        response = get_gemini_response_with_context(user_input, df)
        return response # Returns string
    except Exception as e:
        return f"🤖 Gemini Error: {str(e)}. Please ensure your API key is configured."