import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Weather Prediction Analysis",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌦️ Weather Prediction Analysis Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "📊 Data Overview", 
    "📈 Exploratory Data Analysis", 
    "🤖 Model Training & Results",
    "🔮 Make Predictions"
])

# File upload
st.sidebar.markdown("### Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load data function
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path:
        df = pd.read_csv(file_path)
    else:
        st.error("Please upload a CSV file or provide a file path")
        return None
    return df

# Process data function
@st.cache_data
def process_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df_processed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Store original categorical columns for reference
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    return df_processed, cat_cols, label_encoders

# Train model function
@st.cache_data
def train_model(df_processed):
    if 'RainTomorrow' not in df_processed.columns:
        st.error("Target variable 'RainTomorrow' not found in dataset")
        return None, None, None, None, None, None
    
    X = df_processed.drop('RainTomorrow', axis=1)
    y = df_processed['RainTomorrow']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

# Load data
df = None
if uploaded_file is not None:
    df = load_data(uploaded_file=uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
else:
    st.sidebar.info("👆 Please upload a CSV file to get started")

if df is not None:
    # Process data
    df_processed, cat_cols, label_encoders = process_data(df.copy())
    
    # PAGE 1: Data Overview
    if page == "📊 Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            if 'RainTomorrow' in df.columns:
                rain_tomorrow_ratio = df['RainTomorrow'].value_counts(normalize=True)
                if 'Yes' in rain_tomorrow_ratio:
                    st.metric("Rain Tomorrow %", f"{rain_tomorrow_ratio['Yes']*100:.1f}%")
                else:
                    st.metric("Rain Tomorrow %", "N/A")
        
        st.subheader("📋 First 10 Rows")
        st.dataframe(df.head(10))
        
        st.subheader("📊 Dataset Info")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(dtype_df)
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df)
            else:
                st.write("No missing values found! 🎉")
        
        st.subheader("📈 Statistical Summary")
        tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
        
        with tab1:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 0:
                st.dataframe(df[numerical_cols].describe())
            else:
                st.write("No numerical columns found.")
        
        with tab2:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.dataframe(df[categorical_cols].describe())
            else:
                st.write("No categorical columns found.")
    
    # PAGE 2: EDA
    elif page == "📈 Exploratory Data Analysis":
        st.header("📈 Exploratory Data Analysis")
        
        # Target variable distribution
        if 'RainTomorrow' in df.columns:
            st.subheader("🎯 Target Variable Distribution")
            fig = px.pie(
                values=df['RainTomorrow'].value_counts().values,
                names=df['RainTomorrow'].value_counts().index,
                title="Distribution of RainTomorrow"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Numerical features analysis
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            st.subheader("📊 Numerical Features Distribution")
            
            # Histograms
            cols_per_row = 3
            n_rows = (len(numerical_cols) + cols_per_row - 1) // cols_per_row
            
            fig = make_subplots(
                rows=n_rows, 
                cols=cols_per_row,
                subplot_titles=numerical_cols
            )
            
            for i, col in enumerate(numerical_cols):
                row = i // cols_per_row + 1
                col_pos = i % cols_per_row + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=300*n_rows, title_text="Distribution of Numerical Features")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("🔥 Correlation Heatmap")
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix of Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Boxplots for outlier detection
            st.subheader("📦 Outlier Detection (Boxplots)")
            selected_cols = st.multiselect(
                "Select columns for boxplot analysis:",
                numerical_cols,
                default=numerical_cols[:4] if len(numerical_cols) >= 4 else numerical_cols
            )
            
            if selected_cols:
                fig = make_subplots(
                    rows=len(selected_cols), 
                    cols=1,
                    subplot_titles=selected_cols
                )
                
                for i, col in enumerate(selected_cols):
                    fig.add_trace(
                        go.Box(y=df[col], name=col, showlegend=False),
                        row=i+1, col=1
                    )
                
                fig.update_layout(height=200*len(selected_cols), title_text="Boxplots for Outlier Detection")
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("📊 Categorical Features Analysis")
            
            selected_cat_col = st.selectbox("Select a categorical column:", categorical_cols)
            
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f"Distribution of {selected_cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: Model Training
    elif page == "🤖 Model Training & Results":
        st.header("🤖 Model Training & Results")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... Please wait..."):
                model, X_train, X_test, y_train, y_test, y_pred = train_model(df_processed)
                
                if model is not None:
                    st.success("✅ Model trained successfully!")
                    
                    # Store model in session state
                    st.session_state['model'] = model
                    st.session_state['feature_names'] = X_train.columns.tolist()
                    st.session_state['label_encoders'] = label_encoders
                    
                    # Model performance metrics
                    col1, col2, col3 = st.columns(3)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                    
                    with col2:
                        precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
                        st.metric("🎪 Precision", f"{precision:.3f}")
                    
                    with col3:
                        recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
                        st.metric("🔄 Recall", f"{recall:.3f}")
                    
                    # Confusion Matrix
                    st.subheader("🔀 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    st.subheader("📋 Detailed Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df)
                    
                    # Feature Importance
                    st.subheader("🔍 Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Most Important Features"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: Make Predictions
    elif page == "🔮 Make Predictions":
        st.header("🔮 Make Predictions")
        
        if 'model' in st.session_state:
            st.success("✅ Model is ready for predictions!")
            
            st.subheader("Enter Weather Data:")
            
            # Create input fields for features
            feature_names = st.session_state['feature_names']
            input_data = {}
            
            # Get sample data for reference
            sample_row = df_processed.iloc[0]
            
            # Create input fields in columns
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    if feature in df.select_dtypes(include=['float64', 'int64']).columns:
                        # Numerical input
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        default_val = float(sample_row[feature])
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"input_{feature}"
                        )
                    else:
                        # Categorical input (already encoded in processed data)
                        min_val = int(df_processed[feature].min())
                        max_val = int(df_processed[feature].max())
                        default_val = int(sample_row[feature])
                        input_data[feature] = st.number_input(
                            f"{feature} (encoded)",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"input_{feature}"
                        )
            
            if st.button("🎯 Predict", type="primary"):
                # Create prediction input
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_df)[0]
                prediction_proba = st.session_state['model'].predict_proba(input_df)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("🌧️ Prediction: It will rain tomorrow!")
                    else:
                        st.info("☀️ Prediction: It will NOT rain tomorrow!")
                
                with col2:
                    st.write("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame({
                        'Outcome': ['No Rain', 'Rain'],
                        'Probability': prediction_proba
                    })
                    st.dataframe(prob_df)
                
                # Probability visualization
                fig = px.bar(
                    prob_df,
                    x='Outcome',
                    y='Probability',
                    title="Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Please train the model first in the 'Model Training & Results' section.")

else:
    st.info("👆 Please upload a CSV file to start the analysis!")
    
    # Show sample data format
    st.subheader("📋 Expected Data Format")
    st.write("Your CSV file should contain weather data with a 'RainTomorrow' target column.")
    
    sample_data = {
        'MinTemp': [13.4, 7.4, 12.9],
        'MaxTemp': [22.9, 25.1, 25.7],
        'Rainfall': [0.6, 0.0, 0.0],
        'WindSpeed': [44, 44, 46],
        'Humidity': [71, 44, 38],
        'Pressure': [1007.7, 1010.6, 1007.6],
        'Temperature': [16.9, 16.9, 21.0],
        'RainTomorrow': ['No', 'No', 'No']
    }
    sample_df = pd.DataFrame(sample_data)
    st.write("**Sample data structure:**")
    st.dataframe(sample_df)

# Footer
st.markdown("---")
st.markdown("**🌦️ Weather Prediction Dashboard** | Built with Streamlit")
