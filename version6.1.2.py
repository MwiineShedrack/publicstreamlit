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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
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
        return df.fillna(df.mean())
    elif method == "Fill with Median":
        return df.fillna(df.median())
    elif method == "Fill with Mode":
        return df.fillna(df.mode().iloc[0])
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
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].mean(), df[col])
        elif method == "Replace with Median":
            df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median(), df[col])
    return df

# Function for feature engineering
def feature_engineering(df):
    df['New_Feature_Sum'] = df.select_dtypes(include=[np.number]).sum(axis=1)
    df['New_Feature_Product'] = df.select_dtypes(include=[np.number]).prod(axis=1)
    return df

# Function for regression analysis
def regression_analysis(df, x_columns, y_column):
    X = df[x_columns]
    y = df[y_column]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

# Hypothesis Testing: t-test, ANOVA, Chi-square
def perform_t_test(df, col1, col2):
    stat, p_value = stats.ttest_ind(df[col1], df[col2], nan_policy='omit')
    return stat, p_value

def perform_anova(df, group_col, value_col):
    groups = [df[df[group_col] == group][value_col] for group in df[group_col].unique()]
    stat, p_value = stats.f_oneway(*groups)
    return stat, p_value

def perform_chi_square(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value

# Function for Machine Learning Models
def train_ml_model(df, x_columns, y_column, model_type="Linear Regression"):
    X = df[x_columns]
    y = df[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor()
    elif model_type == "Random Forest":
        model = RandomForestRegressor()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Performance Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, y_test, y_pred, rmse, r2

# Function for K-Means Clustering
def kmeans_clustering(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=[np.number]))
    return df, kmeans

# Function to plot K-Means Clusters
def plot_kmeans_clusters(df, num_clusters):
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color='Cluster', title=f"K-Means Clustering with {num_clusters} Clusters")
    return fig

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Function to detect strong linear relationships
def detect_strong_linear_relationships(df, threshold=0.8):
    corr_matrix = df.corr()
    strong_corr_pairs = []
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix[col1][col2]) >= threshold:
                strong_corr_pairs.append((col1, col2, corr_matrix[col1][col2]))
    return strong_corr_pairs

# Function to detect non-linear relationships using Decision Tree Regressor
def detect_non_linear_relationships(df, target_column, threshold=0.1):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    linear_rmse = np.sqrt(mean_squared_error(y, y_pred_linear))
    
    # Decision tree model (non-linear)
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X, y)
    y_pred_tree = tree_model.predict(X)
    tree_rmse = np.sqrt(mean_squared_error(y, y_pred_tree))
    
    return linear_rmse, tree_rmse

# Function to download processed data
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data
    
# Streamlit App
st.title("Mwiine's Data Analytics App")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())
    
    # Missing values analysis
    st.write("### Missing Values Analysis")
    st.dataframe(missing_values_analysis(df))
    
    # Handle missing values
    missing_value_method = st.selectbox("Choose how to handle missing values", 
                                       ["None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value"], key="missing_values")
    custom_fill_value = None
    if missing_value_method == "Fill with Custom Value":
        custom_fill_value = st.text_input("Enter Custom Value", key="custom_fill_value")
    
    if missing_value_method != "None":
        df = handle_missing_values(df, missing_value_method, custom_fill_value)
        st.write("### Data After Handling Missing Values")
        st.dataframe(df.head())
    
    # Outlier analysis
    st.write("### Outlier Detection")
    st.dataframe(detect_outliers(df))
    
    # Handle outliers
    outlier_method = st.selectbox("Choose how to handle outliers", ["None", "Remove Outliers", "Replace with Mean", "Replace with Median"], key="outlier_handling")
    if outlier_method != "None":
        df = handle_outliers(df, outlier_method)
        st.write("### Data After Handling Outliers")
        st.dataframe(df.head())
    
    # Feature engineering
    st.write("### Feature Engineering")
    df = feature_engineering(df)
    st.write("### Data After Feature Engineering")
    st.dataframe(df.head())
    
    # Descriptive statistics
    st.write("### Descriptive Statistics")
    st.dataframe(df.describe())
    
    # Correlation analysis
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Strong Linear Relationships Detection
    st.write("### Detect Strong Linear Relationships")
    linear_relationships = detect_strong_linear_relationships(df)
    if linear_relationships:
        st.write(f"Strong Linear Relationships (Correlation > 0.8):")
        for col1, col2, corr_value in linear_relationships:
            st.write(f"{col1} and {col2}: Correlation = {corr_value:.2f}")
    else:
        st.write("No strong linear relationships found.")

    #Strong non linear relationships
    def detect_non_linear_relationships(df, target):
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import pandas as pd
        
        df = df.select_dtypes(exclude=['datetime64'])
        
        X = df.drop(columns=[target])  # Exclude target from features
        y = df[target]
        
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        X = X.dropna()
        y = y.loc[X.index]  # Keep only rows where X has values
        
        linear_model = LinearRegression()
        tree_model = DecisionTreeRegressor()
        
        linear_model.fit(X, y)
        tree_model.fit(X, y)
        
        linear_pred = linear_model.predict(X)
        tree_pred = tree_model.predict(X)
        
        linear_rmse = np.sqrt(mean_squared_error(y, linear_pred))
        tree_rmse = np.sqrt(mean_squared_error(y, tree_pred))
        
        return linear_rmse, tree_rmse

    # Regression Analysis
    st.write("### Regression Analysis")
    target = st.selectbox("Select Dependent Variable (Y)", df.columns, key="regression_target")
    predictors = st.multiselect("Select Independent Variables (X)", [col for col in df.columns if col != target], key="regression_predictors")
    
    if st.button("Run Regression Analysis"):
        reg_summary = regression_analysis(df, predictors, target)
        st.text(reg_summary)
    
    # Hypothesis Testing
    st.write("### Hypothesis Testing")
    test_type = st.selectbox("Select Hypothesis Test", ["T-Test", "ANOVA", "Chi-Square"], key="hypothesis_test")

    if test_type == "T-Test":
        col1 = st.selectbox("Select Column 1", df.columns, key="t_test_col1")
        col2 = st.selectbox("Select Column 2", df.columns, key="t_test_col2")
        if st.button("Run T-Test"):
            stat, p_value = perform_t_test(df, col1, col2)
            st.write(f"T-Statistic: {stat}, P-Value: {p_value}")
    elif test_type == "ANOVA":
        group_col = st.selectbox("Select Group Column", df.columns, key="anova_group_col")
        value_col = st.selectbox("Select Value Column", df.columns, key="anova_value_col")
        if st.button("Run ANOVA"):
            stat, p_value = perform_anova(df, group_col, value_col)
            st.write(f"F-Statistic: {stat}, P-Value: {p_value}")
    elif test_type == "Chi-Square":
        col1 = st.selectbox("Select Column 1", df.columns, key="chi_square_col1")
        col2 = st.selectbox("Select Column 2", df.columns, key="chi_square_col2")
        if st.button("Run Chi-Square Test"):
            chi2_stat, p_value = perform_chi_square(df, col1, col2)
            st.write(f"Chi-Square Stat: {chi2_stat}, P-Value: {p_value}")

    # Machine Learning Models
    st.write("### Machine Learning Models")
    model_type = st.selectbox("Select Model Type", ["Linear Regression", "Decision Tree", "Random Forest"], key="ml_model_type")
    target = st.selectbox("Select Dependent Variable (Y)", df.columns, key="ml_target")
    predictors = st.multiselect("Select Independent Variables (X)", [col for col in df.columns if col != target], key="ml_predictors")

    if st.button("Train Model"):
        model, y_test, y_pred, rmse, r2 = train_ml_model(df, predictors, target, model_type)
        st.write(f"RMSE: {rmse}")
        st.write(f"R^2: {r2}")
        st.write(f"Model Coefficients: {model.coef_}")
        # Provide an option to save the trained model
        if st.button("Save Model"):
            filename = "trained_model.pkl"  # Name of the file to save the model
            save_model(model, filename)
            st.success(f"Model saved as {filename}")
        
    # K-Means Clustering
    st.write("### K-Means Clustering")
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3, key="kmeans_clusters")
    if st.button("Run K-Means Clustering"):
        df_clustered, kmeans = kmeans_clustering(df, num_clusters)
        st.write(f"Cluster Centers: {kmeans.cluster_centers_}")
        st.write("### Data with Clusters")
        st.dataframe(df_clustered.head())
        fig = plot_kmeans_clusters(df_clustered, num_clusters)
        st.plotly_chart(fig)
    
    # Download processed data
    processed_data = convert_df_to_csv(df)
    st.download_button("Download Processed Data", processed_data, "processed_data.csv", "text/csv")
