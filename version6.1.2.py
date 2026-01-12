import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from io import BytesIO
import pickle

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

# Function to analyze missing values
def missing_values_analysis(df):
    return df.isnull().sum()

# Function to handle missing values
def handle_missing_values(df, method, fill_value=None):
    if method == "Drop Rows":
        return df.dropna()
    elif method == "Drop Columns":
        return df.dropna(axis=1)
    elif method == "Fill with Mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif method == "Fill with Median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    elif method == "Fill with Mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else np.nan)
        return df
    elif method == "Fill with Custom Value":
        return df.fillna(fill_value)
    return df

# Function to detect outliers using IQR
def detect_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
    return pd.DataFrame.from_dict(outliers, orient='index', columns=['Outlier Count'])

# Function to handle outliers using IQR
def handle_outliers(df, method):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if method == "Remove Outliers":
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == "Replace with Mean":
            mean_val = df[col].mean()
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = mean_val
        elif method == "Replace with Median":
            median_val = df[col].median()
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median_val
    return df

# Function for feature engineering
def feature_engineering(df, sum_feature=True, product_feature=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if sum_feature:
        df['New_Feature_Sum'] = df[numeric_cols].sum(axis=1)
    if product_feature:
        # Avoid overflow by using log product or skip if zeros
        if (df[numeric_cols] == 0).any().any():
            st.warning("Product feature skipped due to zeros in data.")
        else:
            df['New_Feature_Product'] = df[numeric_cols].prod(axis=1)
    return df

# Function for regression analysis
def regression_analysis(df, x_columns, y_column):
    try:
        X = df[x_columns]
        y = df[y_column]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model.summary()
    except Exception as e:
        return f"Error in regression: {str(e)}"

# Hypothesis Testing: t-test, ANOVA, Chi-square
def perform_t_test(df, col1, col2):
    try:
        stat, p_value = stats.ttest_ind(df[col1].dropna(), df[col2].dropna())
        return stat, p_value
    except:
        return None, None

def perform_anova(df, group_col, value_col):
    try:
        groups = [df[df[group_col] == group][value_col].dropna() for group in df[group_col].unique()]
        stat, p_value = stats.f_oneway(*groups)
        return stat, p_value
    except:
        return None, None

def perform_chi_square(df, col1, col2):
    try:
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2_stat, p_value
    except:
        return None, None

# Function for Machine Learning Models
def train_ml_model(df, x_columns, y_column, model_type="Linear Regression", task_type="Regression"):
    try:
        X = df[x_columns]
        y = df[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type == "Regression":
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_type == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            metrics = {"RMSE": rmse, "R2": r2}
            if hasattr(model, 'coef_'):
                coef = model.coef_
            else:
                coef = None
        else:  # Classification
            if model_type == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            else:
                raise ValueError("Linear Regression not suitable for classification.")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_mat = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            metrics = {"Accuracy": accuracy, "Confusion Matrix": conf_mat, "Classification Report": class_report}
            coef = None
        
        return model, y_test, y_pred, metrics, coef
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

# Function for K-Means Clustering
def kmeans_clustering(df, features, num_clusters):
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[features])
        return df, kmeans
    except:
        return df, None

# Function to plot K-Means Clusters (improved to select x and y)
def plot_kmeans_clusters(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color='Cluster', title="K-Means Clustering")
    return fig

# Function to detect strong linear relationships
def detect_strong_linear_relationships(df, threshold=0.8):
    corr_matrix = df.corr(numeric_only=True)
    strong_corr_pairs = set()
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 < col2 and abs(corr_matrix[col1][col2]) >= threshold:
                strong_corr_pairs.add((col1, col2, corr_matrix[col1][col2]))
    return list(strong_corr_pairs)

# Function to detect non-linear relationships
def detect_non_linear_relationships(df, target_column, threshold=0.1):
    try:
        df_numeric = df.select_dtypes(include=[np.number]).dropna()
        if target_column not in df_numeric.columns:
            return None, None
        X = df_numeric.drop(columns=[target_column])
        y = df_numeric[target_column]
        if X.empty or len(X) < 2:
            return None, None
        
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        y_pred_linear = linear_model.predict(X)
        linear_rmse = np.sqrt(mean_squared_error(y, y_pred_linear))
        
        tree_model = DecisionTreeRegressor(random_state=42)
        tree_model.fit(X, y)
        y_pred_tree = tree_model.predict(X)
        tree_rmse = np.sqrt(mean_squared_error(y, y_pred_tree))
        
        return linear_rmse, tree_rmse
    except:
        return None, None

# Function to download processed data
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

# Streamlit App
st.title("Advanced Data Analytics Dashboard")

# Use session state to persist data
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader(" To begin analysis, please upload your desired CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    st.session_state.df = load_data(uploaded_file)
    st.success("File uploaded successfully!")

if st.session_state.df is not None:
    df = st.session_state.df.copy()  # Work on a copy to avoid modifying original

    # Use tabs for better UI organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Preview & Cleaning", "Exploratory Analysis", "Feature Engineering", "Statistical Tests", "Machine Learning", "Clustering"])

    with tab1:
        st.header("Data Preview")
        st.dataframe(df.head(10))
        
        st.header("Missing Values Analysis")
        st.dataframe(missing_values_analysis(df))
        
        missing_value_method = st.selectbox("Handle Missing Values", 
                                            ["None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"])
        custom_fill_value = None
        if missing_value_method == "Fill with Custom Value":
            custom_fill_value = st.text_input("Custom Value")
        
        if missing_value_method != "None" and st.button("Apply Missing Value Handling"):
            df = handle_missing_values(df, missing_value_method, custom_fill_value)
            st.session_state.df = df
            st.success("Missing values handled!")
            st.dataframe(df.head())
        
        st.header("Outlier Detection")
        st.dataframe(detect_outliers(df))
        
        outlier_method = st.selectbox("Handle Outliers", ["None", "Remove Outliers", "Replace with Mean", "Replace with Median"])
        if outlier_method != "None" and st.button("Apply Outlier Handling"):
            df = handle_outliers(df, outlier_method)
            st.session_state.df = df
            st.success("Outliers handled!")
            st.dataframe(df.head())

    with tab2:
        st.header("Descriptive Statistics")
        st.dataframe(df.describe())
        
        st.header("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        st.header("Strong Linear Relationships (Corr > 0.8)")
        linear_relationships = detect_strong_linear_relationships(df)
        if linear_relationships:
            for col1, col2, corr_value in linear_relationships:
                st.write(f"{col1} and {col2}: {corr_value:.2f}")
        else:
            st.write("No strong linear relationships found.")
        
        st.header("Non-Linear Relationships Detection")
        target_nonlin = st.selectbox("Select Target for Non-Linear Check", df.columns)
        if st.button("Check Non-Linear Relationships"):
            linear_rmse, tree_rmse = detect_non_linear_relationships(df, target_nonlin)
            if linear_rmse is not None:
                st.write(f"Linear RMSE: {linear_rmse:.2f}")
                st.write(f"Tree RMSE: {tree_rmse:.2f}")
                if tree_rmse < linear_rmse * 0.9:
                    st.write("Potential non-linear relationships detected (Tree model performs better).")
                else:
                    st.write("Relationships appear mostly linear.")
        
        # Added visualizations
        st.header("Visualizations")
        viz_type = st.selectbox("Select Visualization Type", ["Histogram", "Box Plot", "Scatter Plot"])
        if viz_type == "Histogram":
            col = st.selectbox("Select Column", df.columns)
            fig = px.histogram(df, x=col)
            st.plotly_chart(fig)
        elif viz_type == "Box Plot":
            col = st.selectbox("Select Column", df.columns)
            fig = px.box(df, y=col)
            st.plotly_chart(fig)
        elif viz_type == "Scatter Plot":
            x_col = st.selectbox("X Column", df.columns)
            y_col = st.selectbox("Y Column", df.columns)
            fig = px.scatter(df, x=x_col, y=y_col)
            st.plotly_chart(fig)

    with tab3:
        st.header("Feature Engineering")
        add_sum = st.checkbox("Add Sum Feature", value=True)
        add_product = st.checkbox("Add Product Feature (Careful with zeros/large numbers)")
        if st.button("Apply Feature Engineering"):
            df = feature_engineering(df, sum_feature=add_sum, product_feature=add_product)
            st.session_state.df = df
            st.success("Features added!")
            st.dataframe(df.head())

    with tab4:
        st.header("Regression Analysis")
        reg_target = st.selectbox("Dependent Variable (Y)", df.columns)
        reg_predictors = st.multiselect("Independent Variables (X)", [col for col in df.columns if col != reg_target])
        if st.button("Run Regression"):
            if reg_predictors:
                reg_summary = regression_analysis(df, reg_predictors, reg_target)
                st.text(reg_summary)
            else:
                st.warning("Select at least one predictor.")
        
        st.header("Hypothesis Testing")
        test_type = st.selectbox("Test Type", ["T-Test", "ANOVA", "Chi-Square"])
        if test_type == "T-Test":
            col1 = st.selectbox("Column 1", df.columns)
            col2 = st.selectbox("Column 2", df.columns)
            if st.button("Run T-Test"):
                stat, p_value = perform_t_test(df, col1, col2)
                if stat is not None:
                    st.write(f"T-Statistic: {stat:.2f}, P-Value: {p_value:.4f}")
        elif test_type == "ANOVA":
            group_col = st.selectbox("Group Column", df.columns)
            value_col = st.selectbox("Value Column", df.columns)
            if st.button("Run ANOVA"):
                stat, p_value = perform_anova(df, group_col, value_col)
                if stat is not None:
                    st.write(f"F-Statistic: {stat:.2f}, P-Value: {p_value:.4f}")
        elif test_type == "Chi-Square":
            col1 = st.selectbox("Column 1", df.columns)
            col2 = st.selectbox("Column 2", df.columns)
            if st.button("Run Chi-Square"):
                chi2_stat, p_value = perform_chi_square(df, col1, col2)
                if chi2_stat is not None:
                    st.write(f"Chi-Square: {chi2_stat:.2f}, P-Value: {p_value:.4f}")

    with tab5:
        st.header("Machine Learning Models")
        task_type = st.radio("Task Type", ["Regression", "Classification"])
        model_type = st.selectbox("Model Type", ["Linear Regression" if task_type == "Regression" else "", "Decision Tree", "Random Forest"])
        ml_target = st.selectbox("Target Variable (Y)", df.columns)
        ml_predictors = st.multiselect("Features (X)", [col for col in df.columns if col != ml_target])
        if st.button("Train Model"):
            if ml_predictors and model_type:
                model, y_test, y_pred, metrics, coef = train_ml_model(df, ml_predictors, ml_target, model_type, task_type)
                if model:
                    st.subheader("Model Performance")
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            st.write(f"{key}:")
                            st.write(value)
                        elif isinstance(value, dict):
                            st.write(f"{key}:")
                            st.json(value)
                        else:
                            st.write(f"{key}: {value:.4f}")
                    if coef is not None:
                        st.write("Coefficients:", coef)
                    if st.button("Save Trained Model"):
                        save_model(model, "trained_model.pkl")
                        st.success("Model saved!")
            else:
                st.warning("Select predictors and model type.")

    with tab6:
        st.header("K-Means Clustering")
        cluster_features = st.multiselect("Select Features for Clustering", df.select_dtypes(include=[np.number]).columns)
        num_clusters = st.slider("Number of Clusters", 2, 10, 3)
        if st.button("Run K-Means"):
            if cluster_features:
                df, kmeans = kmeans_clustering(df, cluster_features, num_clusters)
                st.session_state.df = df
                if kmeans:
                    st.write("Cluster Centers:")
                    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features))
                    st.dataframe(df.head())
                    
                    st.subheader("Cluster Visualization")
                    if len(cluster_features) >= 2:
                        x_col = st.selectbox("X Axis", cluster_features)
                        y_col = st.selectbox("Y Axis", cluster_features)
                        fig = plot_kmeans_clusters(df, x_col, y_col)
                        st.plotly_chart(fig)
                    else:
                        st.warning("Need at least 2 features for 2D plot.")
            else:
                st.warning("Select features.")

    # Download at the end
    st.header("Download Processed Data")
    processed_data = convert_df_to_csv(st.session_state.df)
    st.download_button("Download CSV", processed_data, "processed_data.csv", "text/csv")
else:
    st.info("Upload a file to get started.")
