import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import base64


# Maximum number of unique values a categorical feature can have to be considered for one-hot encoding.
# Features with more unique values than this will be handled differently (e.g., dropped or label encoded)
# to prevent excessive dimensionality.
MAX_CATEGORICAL_FEATURES_FOR_OHE = 10

# Plotly template for styling charts. Can be set to "plotly_white" or other available templates.
PLOTLY_TEMPLATE = "plotly_dark" # Consider making this configurable via Streamlit theme or user option.

# === Constants ===
TEMP_PLOT_DIR = "static_plots"

# Ensure directory exists
os.makedirs(TEMP_PLOT_DIR, exist_ok=True)

def generate_data_summary(df: pd.DataFrame) -> str:
    """
    Generates a comprehensive summary of the DataFrame.

    Args:
        df: The input Pandas DataFrame.

    Returns:
        A string containing a multi-faceted summary of the DataFrame,
        including shape, data types, head, descriptive statistics,
        missing values, memory usage, and df.info().
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    summary = f"""
### 📋 Full Data Summary:

**1. Shape of the Dataset:**
   - Rows: {df.shape[0]}
   - Columns: {df.shape[1]}

**2. Data Types:**
```
{df.dtypes.to_string()}
```

**3. First 5 Rows (Head):**
```
{df.head().to_markdown(index=False)}
```

**4. Descriptive Statistics (for numerical columns):**
```
{df.describe().to_markdown()}
```

**5. Missing Values (sum per column):**
"""
    missing_values = df.isnull().sum()
    # Filter to show only columns that actually have missing values.
    missing_values_str = missing_values[missing_values > 0].to_string()
    if not missing_values_str.strip() or missing_values[missing_values > 0].empty:
        summary += "   - ✅ No missing values found.\n"
    else:
        summary += f"```\n{missing_values_str}\n```\n"

    summary += f"""
**6. Memory Usage:**
   - {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB

**7. Column Information (from df.info()):**
```
{info_str}
```
"""
    return summary.strip()

def extract_column_names(text: str, df_columns: list, max_cols: int = 1) -> list:
    """
    Extracts one or more column names from a given text string,
    prioritizing quoted names and then direct word matches against DataFrame columns.

    Args:
        text: The user input string.
        df_columns: A list of column names in the DataFrame.
        max_cols: The maximum number of column names to extract.

    Returns:
        A list of extracted column names (original casing) up to max_cols.
    """
    text_lower = text.lower()
    # Create a mapping from lowercase column names to original casing for accurate return.
    original_df_columns_map = {col.lower(): col for col in df_columns}
    df_columns_lower = list(original_df_columns_map.keys())

    extracted_cols = []

    # 1. Regex for quoted column names (single or double quotes)
    # This pattern finds content within quotes.
    quoted_cols_match = re.findall(r"'(.*?)'|\"(.*?)\"", text_lower)
    for q_col_tuple in quoted_cols_match:
        # q_col_tuple will be like ('content', '') or ('', 'content')
        q_col = next((s for s in q_col_tuple if s), None) # Get the non-empty string
        if q_col and q_col in df_columns_lower:
            original_col_name = original_df_columns_map[q_col]
            if original_col_name not in extracted_cols: # Avoid duplicates
                extracted_cols.append(original_col_name)
                if len(extracted_cols) == max_cols:
                    return extracted_cols

    # 2. If not enough columns found with quotes, try matching words from input against column names.
    # This is a broader search.
    words_in_text = set(re.findall(r'\b\w+\b', text_lower)) # Unique words from input

    for col_l in df_columns_lower:
        original_col_name = original_df_columns_map[col_l]
        if original_col_name in extracted_cols: # Skip if already found
            continue

        # Check if the lowercase column name (potentially multi-word) is present in the input text.
        # This helps catch column names with spaces if they weren't quoted.
        if col_l in text_lower:
            extracted_cols.append(original_col_name)
            if len(extracted_cols) == max_cols:
                return extracted_cols
            continue # Move to next df_column_lower

        # Check if any word from the column name (if it's multi-word) is in the input text words.
        # Or if any input text word is part of the column name.
        col_name_parts = set(col_l.split())
        if words_in_text.intersection(col_name_parts):
            extracted_cols.append(original_col_name)
            if len(extracted_cols) == max_cols:
                return extracted_cols

    return extracted_cols


def get_column_stats(user_input: str, df: pd.DataFrame) -> str:
    """
    Calculates and returns descriptive statistics for a specified column.
    Differentiates between numerical and categorical columns for relevant stats.

    Args:
        user_input: The user's query string.
        df: The input Pandas DataFrame.

    Returns:
        A string containing formatted statistics for the identified column,
        or an error message if the column is not found or not identifiable.
    """
    cols_found = extract_column_names(user_input, df.columns, max_cols=1)
    if not cols_found:
        return "❌ Could not identify a valid column name in your request. Please specify clearly, e.g., 'stats for column_name' or 'mean of Age'."
    col = cols_found[0]

    if col not in df.columns:
         return f"❌ Column '{col}' not found in the dataset."

    column_data = df[col]

    if pd.api.types.is_numeric_dtype(column_data):
        desc = column_data.describe()
        mode_series = column_data.mode()
        # Handle cases where mode might be empty (e.g., all unique values in a float column)
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A (no distinct mode)"
        skewness = column_data.skew()
        kurt = column_data.kurtosis()

        return f"""**📊 Statistics for Numerical Column: `{col}`**
- Count:    {desc.get('count', 0):.0f}
- Mean:     {desc.get('mean', float('nan')):.2f}
- Median:   {desc.get('50%', float('nan')):.2f} (50th percentile)
- Mode:     {mode_val}
- Std Dev:  {desc.get('std', float('nan')):.2f}
- Min:      {desc.get('min', float('nan')):.2f}
- Max:      {desc.get('max', float('nan')):.2f}
- 25th Pctl:{desc.get('25%', float('nan')):.2f}
- 75th Pctl:{desc.get('75%', float('nan')):.2f}
- Skewness: {skewness:.2f}
- Kurtosis: {kurt:.2f}
"""
    else: # Categorical or other non-numeric types
        desc = column_data.describe() # For object types, describe() gives count, unique, top, freq
        unique_vals = column_data.nunique()
        mode_series = column_data.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"

        return f"""**📊 Statistics for Categorical/Object Column: `{col}`**
- Count:        {desc.get('count', 0)}
- Unique Values:{unique_vals}
- Top (Most Frequent): {desc.get('top', 'N/A')}
- Frequency of Top: {desc.get('freq', 'N/A')}
- Mode (Calculated): {mode_val}
"""
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")



def generate_plot_interactive(user_input: str, df: pd.DataFrame) -> dict | str:
    input_lower = user_input.lower()
    fig = None
    message = ""
    static_plot_path = ""

    try:
        cols = extract_column_names(input_lower, df.columns, 2)

        # === PLOTLY INTERACTIVE ===
        if "histogram" in input_lower or "distribution" in input_lower:
            col = cols[0] if cols else None
            if col:
                fig = px.histogram(df, x=col, marginal="box", template=PLOTLY_TEMPLATE)
                message = f"✅ Histogram of `{col}`"
            else:
                return "❓ Specify a column for histogram."

        elif "box plot" in input_lower or "boxplot" in input_lower:
            col = cols[0] if cols else None
            if col:
                fig = px.box(df, y=col, template=PLOTLY_TEMPLATE)
                message = f"✅ Box Plot of `{col}`"
            else:
                return "❓ Specify a column for box plot."

        elif "scatter" in input_lower:
            if len(cols) == 2:
                fig = px.scatter(df, x=cols[0], y=cols[1], trendline="ols", template=PLOTLY_TEMPLATE)
                message = f"✅ Scatter Plot of `{cols[0]}` vs `{cols[1]}`"
            else:
                return "❓ Specify two columns for scatter plot."

        elif "correlation heatmap" in input_lower or "correlation matrix" in input_lower:
            return get_correlation_matrix(df)

        elif "dashboard" in input_lower:
            numeric_cols = df.select_dtypes(include='number').columns[:4]
            fig = make_subplots(rows=2, cols=2, subplot_titles=list(numeric_cols))
            for i, col in enumerate(numeric_cols):
                hist = go.Histogram(x=df[col], name=col)
                fig.add_trace(hist, row=i // 2 + 1, col=i % 2 + 1)
            fig.update_layout(title_text="📊 Interactive Dashboard", template=PLOTLY_TEMPLATE)
            message = "✅ Generated interactive dashboard using Plotly."

        elif "pie chart" in input_lower:
            col = cols[0] if cols else None
            if col and not pd.api.types.is_numeric_dtype(df[col]):
                pie_data = df[col].value_counts().reset_index()
                pie_data.columns = [col, 'count']
                fig = px.pie(pie_data, names=col, values='count', template=PLOTLY_TEMPLATE)
                message = f"✅ Pie Chart of `{col}`"
            else:
                return "❓ Pie chart requires a categorical column."


        elif "line plot" in input_lower or "lineplot" in input_lower:
            if len(cols) == 2:
                fig = px.line(df, x=cols[0], y=cols[1], template=PLOTLY_TEMPLATE)
                message = f"✅ Line Plot of `{cols[0]}` vs `{cols[1]}`"
            else:
                return "❓ Line plot needs two columns."

        # === SEABORN / MATPLOTLIB STATIC ===

        elif "barplot" in input_lower and not "horizontal" in input_lower:
            if len(cols) == 2:
                plt.figure(figsize=(8, 6))
                sns.barplot(x=df[cols[0]], y=df[cols[1]])
                plt.title(f"Barplot of {cols[1]} by {cols[0]}")
                static_plot_path = os.path.join(TEMP_PLOT_DIR, f"barplot_{uuid.uuid4().hex}.png")
                plt.savefig(static_plot_path)
                plt.close()
                base64_img = encode_image_to_base64(static_plot_path)
                return {"type": "static_base64", 
                        "image": base64_img, 
                        "format": "png", 
                        "message": f"✅ Barplot of `{cols[1]}` by `{cols[0]}`"}
            else:
                return "❓ Barplot needs two columns: categorical and numerical."
        
        elif "horizontal bar" in input_lower:
            if len(cols) == 2:
                plt.figure(figsize=(8, 6))
                sns.barplot(y=df[cols[0]], x=df[cols[1]])
                plt.title(f"Horizontal Bar Plot of {cols[0]} by {cols[1]}")
                static_plot_path = os.path.join(TEMP_PLOT_DIR, f"hbarplot_{uuid.uuid4().hex}.png")
                plt.savefig(static_plot_path)
                plt.close()

                base64_img = encode_image_to_base64(static_plot_path)
                return {
                    "type": "static_base64",
                    "image": base64_img,
                    "format": "png",
                    "message": f"✅ Horizontal Bar Plot of `{cols[0]}` by `{cols[1]}`"
                }
            else:
                return "❓ Horizontal bar plot requires a categorical column."

        elif "pairplot" in input_lower:
            pairplot_fig = sns.pairplot(df.select_dtypes(include='number'))
            static_plot_path = os.path.join(TEMP_PLOT_DIR, f"pairplot_{uuid.uuid4().hex}.png")
            pairplot_fig.figure.savefig(static_plot_path)
            plt.close()
            
            base64_img = encode_image_to_base64(static_plot_path)
            return {
                "type": "static_base64",
                "image": base64_img,
                "format": "png",
                "message": "✅ Seaborn Pairplot"
            }

        elif "violin" in input_lower or "violinplot" in input_lower:
            col = cols[0] if cols else None
            if col:
                plt.figure(figsize=(8, 6))
                sns.violinplot(y=df[col])
                plt.title(f"Violin plot of {col}")
                static_plot_path = os.path.join(TEMP_PLOT_DIR, f"violin_{uuid.uuid4().hex}.png")
                plt.savefig(static_plot_path)
                plt.close()
                
                base64_img = encode_image_to_base64(static_plot_path)
                return {
                    "type": "static_base64",
                    "image": base64_img,
                    "format": "png",
                    "message": f"✅ Violin plot of `{col}`"
                }
            else:
                return "❓ Specify a column for violin plot."

        elif "kde" in input_lower or "kdeplot" in input_lower:
            col = cols[0] if cols else None
            if col:
                plt.figure(figsize=(8, 6))
                sns.kdeplot(df[col], fill=True)
                plt.title(f"KDE plot of {col}")
                static_plot_path = os.path.join(TEMP_PLOT_DIR, f"kde_{uuid.uuid4().hex}.png")
                plt.savefig(static_plot_path)
                plt.close()

                base64_img = encode_image_to_base64(static_plot_path)
                return {
                    "type": "static_base64",
                    "image": base64_img,
                    "format": "png",
                    "message": f"✅ KDE plot of `{col}`"
                }   

            else:
                return "❓ Specify a column for KDE plot."

        elif "joint plot" in input_lower:
            if len(cols) == 2:
                joint_fig = sns.jointplot(x=cols[0], y=cols[1], data=df, kind="scatter")
                static_plot_path = os.path.join(TEMP_PLOT_DIR, f"jointplot_{uuid.uuid4().hex}.png")
                joint_fig.figure.savefig(static_plot_path)
                plt.close()
                base64_img = encode_image_to_base64(static_plot_path)
                return {
                    "type": "static_base64",
                    "image": base64_img,
                    "format": "png",
                    "message": f"✅ Joint Plot of `{cols[0]}` vs `{cols[1]}`"
            }
            else:
                return "❓ Joint plot needs two columns."

        else:
            return "❓ Could not determine the plot type. Try specifying keywords like histogram, box plot, pairplot, etc."

        if fig:
            return {"type": "plotly", "fig": fig, "message": message}

    except Exception as e:
        return f"❌ Error generating plot: {str(e)}"


def get_correlation_matrix(df: pd.DataFrame) -> dict | str:
    """
    Computes and returns a correlation matrix heatmap for numerical columns.

    Args:
        df: The input Pandas DataFrame.

    Returns:
        A dictionary containing the Plotly figure for the heatmap and a message,
        or a string message if not enough numerical columns are available.
    """
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.shape[1] < 2:
        return "⚠️ Not enough numerical columns to compute a correlation matrix (at least 2 are required)."

    corr_matrix = numerical_df.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", # Display correlation values on the heatmap
                    aspect="auto", # Adjust aspect ratio automatically
                    title="Correlation Heatmap of Numerical Features",
                    color_continuous_scale='RdBu_r', # Red-Blue diverging color scale
                    template=PLOTLY_TEMPLATE)
    return {"type": "plotly", "fig": fig, "message": "✅ Correlation heatmap displayed."}


def extract_target_column(user_input: str, df_columns: pd.Index) -> str | None:
    """
    Extracts the target column for machine learning tasks from user input.
    It looks for explicit mentions like "predict 'column_name'" or "target is 'column_name'",
    or infers from context. Falls back to the last column if no clear target is found.

    Args:
        user_input: The user's query string.
        df_columns: A Pandas Index object containing the column names of the DataFrame.

    Returns:
        The identified target column name (original casing), or the last column name as a fallback.
        Returns None if df_columns is empty.
    """
    if df_columns.empty:  # Corrected check: use .empty for Pandas Index
        return None

    user_input_lower = user_input.lower()
    df_columns_lower_map = {col.lower(): col for col in df_columns}

    # Patterns to find explicitly mentioned target columns.
    # Looks for keywords like "predict", "target is", "for column" followed by a potential column name.
    patterns = [
        r"predict\s+(?:column\s+)?(['\"]?(.*?)['\"]?)\b",
        r"target\s*(?:is|=)\s*(?:column\s+)?(['\"]?(.*?)['\"]?)\b",
        r"for\s+column\s+(['\"]?(.*?)['\"]?)\b",
        r"dependent\s*(?:variable\s*(?:is|=)?)?\s*(['\"]?(.*?)['\"]?)\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            # Extract the captured group that contains the column name.
            # It might be group 1 or 2 depending on optional quotes.
            potential_target_in_match = next((g.strip() for g in match.groups() if g and g.strip()), None)
            if potential_target_in_match and potential_target_in_match in df_columns_lower_map:
                return df_columns_lower_map[potential_target_in_match]

    # If no explicit pattern matched, use the general column extraction logic,
    # looking for any column name mentioned in the input.
    # This is less precise for target identification but can be a fallback.
    cols_from_general_extraction = extract_column_names(user_input_lower, list(df_columns), 1) # Convert Index to list
    if cols_from_general_extraction:
        return cols_from_general_extraction[0]

    # As a final fallback, assume the last column is the target.
    return df_columns[-1]


def train_and_evaluate_model(user_input: str, df: pd.DataFrame) -> str:
    """
    Trains a simple machine learning model (Regression or Classification)
    based on the user input and DataFrame. Includes basic preprocessing,
    model training, evaluation, and feature importance (if applicable).

    Args:
        user_input: The user's query, used to infer the target variable.
        df: The input Pandas DataFrame.

    Returns:
        A string containing the model training results and evaluation metrics,
        or an error message if training fails.
    """
    try:
        target_col = extract_target_column(user_input, df.columns)
        if not target_col:
            return "❌ Could not determine the target variable for model training. Please specify it, e.g., 'train model to predict ColumnName'."

        df_processed = df.copy()
        df_processed.dropna(subset=[target_col], inplace=True) # Drop rows where target is NaN

        if df_processed.empty:
            return f"❌ Dataset became empty after removing rows with missing target values in '{target_col}'. Cannot train model."

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

        # Identify and handle high-cardinality categorical features (too many unique values for OHE)
        high_cardinality_cats = []
        categorical_features_for_ohe = []
        for col in categorical_features:
            if X[col].nunique() > MAX_CATEGORICAL_FEATURES_FOR_OHE:
                high_cardinality_cats.append(col)
            else:
                categorical_features_for_ohe.append(col)
        
        # Define preprocessing steps for numerical and categorical features
        # Numerical: Impute missing with median, then scale.
        # Categorical: Impute missing with most frequent, then one-hot encode.
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create a column transformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features_for_ohe)
            ],
            remainder='drop' # Drop columns not specified (e.g., high cardinality cats or ID columns)
        )

        # Determine model type based on the target variable's nature
        label_encoder = None # Initialize LabelEncoder for classification tasks
        if pd.api.types.is_numeric_dtype(y):
            # Heuristic: if a numeric target has few unique values, treat as classification.
            if y.nunique() > 15 and y.nunique() / len(y) > 0.1: # Check if continuous enough
                model_type = 'regression'
                model = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
                scoring_metric_name = 'R² Score'
                secondary_metric_name = 'Mean Squared Error (MSE)'
            else: # Treat as classification (e.g., 0, 1, 2 or few distinct values)
                model_type = 'classification'
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y) # Encode numeric target to 0, 1, ...
                model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
                scoring_metric_name = 'Accuracy'
                secondary_metric_name = 'Classification Report'
        else: # Categorical target
            model_type = 'classification'
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y) # Encode categorical target to 0, 1, ...
            model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1, min_samples_leaf=5)
            scoring_metric_name = 'Accuracy'
            secondary_metric_name = 'Classification Report'

        if len(X) < 10 or (model_type == 'classification' and len(np.unique(y)) < 2):
             return f"❌ Insufficient data or too few unique classes in target '{target_col}' to train a meaningful {model_type} model after preprocessing."

        # Create the full pipeline including preprocessing and the model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        if y.shape[0] < 5: # Need at least a few samples for splitting
            return "❌ Not enough samples to split into training and testing sets after initial preprocessing."

        test_size = 0.25
        # Stratify for classification if possible to maintain class proportions
        stratify_param = y if model_type == 'classification' and len(np.unique(y)) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_param)
        except ValueError as e: # Catch errors from train_test_split (e.g., too few samples in a class for stratification)
            if "The least populated class" in str(e) or "n_splits=" in str(e):
                 # Fallback to non-stratified split if stratification fails
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            else:
                raise e # Re-raise other ValueErrors

        if X_train.empty or X_test.empty:
            return "❌ Training or testing set became empty after splitting. This usually means there's very little data."

        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)

        # --- Constructing the Result String ---
        result_str = f"### 🤖 Model Training Results for Target: `{target_col}`\n"
        result_str += f"- **Model Type:** {model.__class__.__name__} ({model_type})\n"
        
        # Get actual feature names after preprocessing (can be complex with OHE)
        try:
            feature_names_out = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
            result_str += f"- **Number of Features Used by Model (after preprocessing):** {len(feature_names_out)}\n"
        except Exception:
            # If get_feature_names_out fails or preprocessor isn't set up as expected
            result_str += "- *Number of features used could not be determined post-preprocessing.*\n"

        if high_cardinality_cats:
            result_str += f"- **Categorical features dropped due to high cardinality (> {MAX_CATEGORICAL_FEATURES_FOR_OHE} unique values):** {', '.join(high_cardinality_cats)}\n"
        if not numerical_features and not categorical_features_for_ohe:
             result_str += "- **Warning:** No features were selected for preprocessing. The model might not be meaningful.\n"


        if model_type == 'regression':
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            result_str += f"- **{scoring_metric_name}:** {score:.3f}\n"
            result_str += f"- **{secondary_metric_name}:** {mse:.3f}\n"
        else: # Classification
            score = accuracy_score(y_test, y_pred)
            # Ensure target names are strings for classification_report if LabelEncoder was used
            target_names_for_report = [str(cls_name) for cls_name in label_encoder.classes_] if label_encoder else None
            report = classification_report(y_test, y_pred, target_names=target_names_for_report, zero_division=0)
            result_str += f"- **{scoring_metric_name}:** {score:.3f}\n"
            result_str += f"\n**{secondary_metric_name}:**\n```\n{report}\n```\n"

        # Feature Importances (for tree-based models like RandomForest)
        if hasattr(model, 'feature_importances_'):
            try:
                # Get feature names after one-hot encoding from the preprocessor step
                feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10) # Show top 10
                result_str += f"\n**Top Feature Importances (up to 10):**\n```\n{feature_importance_df.to_string(index=False)}\n```\n"
            except Exception as fi_e:
                print(f"Could not get feature importances: {fi_e}") # Log error
                result_str += "- *Feature importances could not be extracted for this pipeline setup.*\n"

        return result_str + "\n\n*Note: This is a simplified model trained for quick analysis. For production, more rigorous feature engineering, hyperparameter tuning, and model selection would be required.*"

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc() # Get the full traceback
        return f"❌ Failed to train model: {str(e)}\n\n<details><summary>Click for technical traceback</summary>\n\n```\n{tb_str}\n```\n</details>"


# Placeholder for a function that might be used if app.py handles plotting directly.
# Currently, generate_plot_interactive returns the figure object.
def get_plot_function_and_cols(user_input: str, df: pd.DataFrame):
    """
    This function is a conceptual placeholder.
    It would parse the user input to determine the plot type and relevant columns,
    returning this information for another part of the application to use for plotting.
    Currently, `generate_plot_interactive` handles both parsing and figure generation.
    """
    # Example parsing logic (would need to be implemented robustly):
    # if "histogram" in user_input.lower():
    #     plot_type = "histogram"
    #     cols = extract_column_names(user_input, df.columns, 1)
    #     return plot_type, cols
    # ... other plot types ...
    pass
