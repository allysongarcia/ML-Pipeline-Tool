import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.compose import ColumnTransformer
import joblib
import shap
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import time
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, 
    mean_squared_error, r2_score, precision_score, recall_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error

# Initialize logging at the top
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs.txt",
    filemode="a"  # "a" to append to the file
)
logger = logging.getLogger(__name__)

# Make sure logging is initialized
logger.info("Logging initialized")

# Page configuration with custom theme
st.set_page_config(
    page_title="ML Pipeline Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAV_STEP_MAPPING = {
    "Data Upload & EDA": 1,
    "Preprocessing": 2,
    "Model Training": 3,
    "Evaluation": 4,
    "Deployment": 5
}

# Custom CSS to enhance UI look and feel
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #4285F4;
        --secondary: #34A853;
        --background: #F9F9F9;
        --text: #202124;
        --light-accent: #E8F0FE;
    }
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Headings */
    h1 {
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 1rem !important;
    }
    h2 {
        font-weight: 600 !important;
        color: #1E3A8A !important;
        margin-top: 2rem !important;
    }
    h3 {
        font-weight: 500 !important;
        color: #2563EB !important;
    }
    
    /* Cards */
    .stCard {
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        padding: 1rem !important;
        background-color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Sidebar */
    [data-testid=stSidebar] {
        background-color: #FAFAFA !important;
        padding: 1rem !important;
    }
    
    /* Success messages */
    .success-message {
        background-color: #D1FAE5 !important;
        border-left: 5px solid #10B981 !important;
        padding: 1rem !important;
        border-radius: 4px !important;
    }
    
    /* Warning messages */
    .warning-message {
        background-color: #FEF3C7 !important;
        border-left: 5px solid #F59E0B !important;
        padding: 1rem !important;
        border-radius: 4px !important;
    }
    
    /* Error messages */
    .error-message {
        background-color: #FEE2E2 !important;
        border-left: 5px solid #EF4444 !important;
        padding: 1rem !important;
        border-radius: 4px !important;
    }
    
    /* Section containers */
    .section-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        margin-bottom: 1.5rem;
    }
    
    /* Metrics display */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    
    /* Data frames */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0 !important;
        border-color: #E5E7EB !important;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress steps */
    .progress-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .progress-step {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        border-bottom: 3px solid #E5E7EB;
        color: #6B7280;
    }
    .progress-step.active {
        border-bottom: 3px solid #2563EB;
        color: #2563EB;
        font-weight: 500;
    }
    .progress-step.completed {
        border-bottom: 3px solid #10B981;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['trained_model', 'X_val', 'y_val_enc', 'label_encoder', 'problem', 'current_step']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'current_step' not in st.session_state or st.session_state.current_step is None:
    st.session_state.current_step = 1

# App header with logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <div style="background-color: #4285F4; width: 80px; height: 80px; border-radius: 50%; display: flex; 
             align-items: center; justify-content: center; margin: 0 auto;">
            <span style="color: white; font-size: 40px;">🧠</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <h1 style="margin-bottom: 0;">ML Pipeline Builder</h1>
    <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
        Build, evaluate, and deploy machine learning models without coding
    </p>
    """, unsafe_allow_html=True)

# Progress tracker
steps = ["Data Upload", "Preprocessing", "Model Configuration", "Training", "Evaluation", "Deployment"]
progress_html = '<div class="progress-container">'
for i, step in enumerate(steps, 1):
    if i < st.session_state.current_step:
        status = "completed"
    elif i == st.session_state.current_step:
        status = "active"
    else:
        status = ""
    progress_html += f'<div class="progress-step {status}">{step}</div>'
progress_html += '</div>'
st.markdown(progress_html, unsafe_allow_html=True)

@st.cache_data
def load_data(f):
    return pd.read_csv(f)

# Sidebar for app navigation and help
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="margin-bottom: 5px;">ML Pipeline Builder</h2>
        <p style="color: #666;">v1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation")

    # # Inside the sidebar code
    # nav_option = st.radio(
    #     "Go To:",
    #     ["Data Upload & EDA", "Preprocessing", "Model Training", "Evaluation", "Deployment"],
    #     label_visibility="collapsed"
    # )

    index = list(NAV_STEP_MAPPING.values()).index(st.session_state.current_step)

    nav_option = st.radio(
        "Go To:",
        ["Data Upload & EDA", "Preprocessing", "Model Training", "Evaluation", "Deployment"],
        index=index,
        label_visibility="collapsed"
    )

    # Update current step based on navigation selection
    if nav_option in NAV_STEP_MAPPING:
        st.session_state.current_step = NAV_STEP_MAPPING[nav_option]
    
    st.markdown("### Help & Resources")
    with st.expander("Quick Start Guide"):
        st.markdown("""
        1. Upload your CSV dataset
        2. Explore data with automatic EDA
        3. Configure preprocessing steps
        4. Select and train your model
        5. Evaluate performance
        6. Download model or make predictions
        """)
    
    with st.expander("Feature Glossary"):
        st.markdown("""
        - **Imputation**: Filling missing values
        - **Feature Scaling**: Normalizing numerical values
        - **Outlier Handling**: Managing extreme values
        - **Validation Set**: Data portion used to test model
        """)
    
    st.markdown("### About")
    st.markdown("""
    This application helps you build machine learning pipelines with no coding required.
    Perfect for data scientists, analysts, and ML beginners.
    """)

# Main content area - wrapped in container for styling
st.markdown('<div class="section-container">', unsafe_allow_html=True)

# Replace all existing if/elif blocks with this structure
current_step = st.session_state.current_step
# FILE UPLOAD AND INITIAL EDA (STEP 1)
# if current_step == 1 or (nav_option == "Data Upload & EDA" and current_step != 1):
if st.session_state.current_step == 1:
    # Data Upload & EDA content
    st.session_state.current_step = 1
    
    st.markdown("""
    <h2><i class="fas fa-file-upload"></i> Data Upload & Exploration</h2>
    <p>Start by uploading your dataset and exploring its characteristics</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], 
                                   help="Upload a CSV file containing your dataset")
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
            st.session_state.df = df
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", f"{df.isna().sum().sum():,}")
        
        tab1, tab2, tab3 = st.tabs(["📊 Preview", "📈 Statistics", "🔍 EDA Report"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
            
        with tab2:
            st.markdown("### Data Summary")
            st.dataframe(df.describe().T, use_container_width=True)
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Column Data Types")
                st.dataframe(dtype_counts, use_container_width=True)
            
            with col2:
                # Missing values analysis
                st.markdown("#### Missing Values")
                missing = df.isna().sum().sort_values(ascending=False)
                missing = missing[missing > 0]
                
                if len(missing) > 0:
                    fig, ax = plt.subplots(figsize=(10, len(missing)/2 + 1))
                    sns.set_style("whitegrid")
                    sns.barplot(x=missing.values, y=missing.index, palette="Blues_r", ax=ax)
                    ax.set_title("Missing values per column")
                    ax.set_xlabel("Count of missing values")
                    st.pyplot(fig)
                else:
                    st.success("No missing values found in the dataset!")
        
        with tab3:
            if st.checkbox("Generate detailed EDA report", help="This may take some time for large datasets"):
                with st.spinner("Generating comprehensive EDA report..."):
                    profile = ProfileReport(df, explorative=True, minimal=True)
                    st_profile_report(profile)
        
        # Next step button
        if st.button("Continue to Preprocessing →", type="primary"):
            st.session_state.current_step = 2
            st.rerun()

# PREPROCESSING (STEP 2)
# elif current_step == 2 or (nav_option == "Preprocessing" and current_step != 2):
elif st.session_state.current_step == 2:
    # Preprocessing content
    st.session_state.current_step = 2
    
    if 'df' not in st.session_state:
        st.error("⚠️ Please upload data first.")
        if st.button("← Go to Data Upload"):
            st.session_state.current_step = 1
            st.rerun()

    else:
        df = st.session_state.df

        st.markdown("""
        <h2><i class="fas fa-filter"></i> Preprocessing & Feature Engineering</h2>
        <p>Configure how your data should be processed before model training</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Missing Value Handling")
            num_impute = st.selectbox(
                "Numerical Imputation", 
                ['mean', 'median', 'constant'],
                help="Strategy to fill missing values in numeric columns"
            )
            cat_impute = st.selectbox(
                "Categorical Imputation", 
                ['most_frequent', 'constant'],
                help="Strategy to fill missing values in categorical columns"
            )
            
            st.markdown("#### Feature Transformation")
            scaling = st.selectbox(
                "Feature Scaling", 
                ['None', 'Standardization', 'Normalization'],
                help="Standardization sets mean=0, std=1. Normalization scales to 0-1 range."
            )

            st.session_state.num_impute = num_impute
            st.session_state.cat_impute = cat_impute
            st.session_state.scaling = scaling
        
        with col2:
            st.markdown("#### Advanced Preprocessing")
            outlier = st.selectbox(
                "Outlier Handling", 
                ['None', 'IQR', 'Z-Score'],
                help="Method to detect and handle outliers in numerical features"
            )
            
            do_poly = st.checkbox(
                "Enable Polynomial Features", 
                help="Create new features from combinations of original features"
            )
            if do_poly:
                poly_degree = st.slider(
                    "Polynomial degree", 
                    1, 3, 1,
                    help="Higher degrees capture more complex relationships but may overfit"
                )
                interaction = st.checkbox(
                    "Interaction only", 
                    help="Only include interaction terms, not polynomial terms"
                )
                
        # Next step button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Go Back", key="preprocessing_back"):
                st.session_state.current_step = 1
                st.experimental_rerun()
        with col2:
            if st.button("Continue to Model Configuration →", type="primary"):
                logger.info(f"BEFORE: session_state.current_step: {st.session_state.get('current_step', 'Not Set')}")
                st.session_state.current_step = 3
                logger.info(f"AFTER: Step-2: session_state.current_step: {st.session_state.current_step}")
                st.rerun()

# MODEL CONFIGURATION (STEP 3)
# elif current_step == 3 or (nav_option == "Model Training" and current_step != 3):
elif st.session_state.current_step == 3:    
    # st.session_state.current_step = 3
    
    if 'df' not in st.session_state:
        st.error("Please upload data first")
        if st.button("← Go to Data Upload"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        df = st.session_state.df

        st.markdown("""
        <h2><i class="fas fa-cogs"></i> Model Configuration & Training</h2>
        <p>Select your model type, target variable and configure training parameters</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Model Selection")
            model_type = st.selectbox(
                "Model Algorithm", 
                ['Random Forest', 'Gradient Boosting', 'SVM'],
                help="Choose the machine learning algorithm to use"
            )
            problem = st.radio(
                "Problem Type", 
                ['Classification', 'Regression'],
                help="Classification predicts categories, regression predicts continuous values"
            )
            st.session_state.problem = problem
        
        with col2:
            st.markdown("#### Target & Feature Selection")
            target = st.selectbox(
                "Target Column (what to predict)", 
                df.columns,
                help="The variable your model will learn to predict"
            )
            
            # Target checks with friendly messages
            if problem == 'Regression' and not pd.api.types.is_numeric_dtype(df[target]):
                st.markdown("""
                <div class="error-message">
                    <strong>⚠️ Warning:</strong> Regression requires a numeric target. 
                    Please choose a numeric column or switch to Classification.
                </div>
                """, unsafe_allow_html=True)
            
            if problem == 'Classification':
                if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                    st.markdown("""
                    <div class="warning-message">
                        <strong>⚠️ Warning:</strong> Continuous target detected with many unique values.
                        Consider Regression instead for better results.
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("#### Select Features for Modeling")
        all_features = df.columns.drop(target).tolist()
        
        # Group features by data type for better organization
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target]
        
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != target]
        
        tab1, tab2 = st.tabs(["Simple Selection", "Advanced Selection"])
        
        with tab1:
            feats = st.multiselect(
                "Features to include in model", 
                all_features, 
                default=all_features,
                help="Select the input variables to use for prediction"
            )
        
        with tab2:
            st.markdown("##### Select by Type")
            use_numeric = st.checkbox("Include numeric features", value=True)
            if use_numeric and numeric_cols:
                selected_numeric = st.multiselect(
                    "Select numeric features",
                    numeric_cols,
                    default=numeric_cols
                )
            else:
                selected_numeric = []
                
            use_categorical = st.checkbox("Include categorical features", value=True) 
            if use_categorical and categorical_cols:
                selected_categorical = st.multiselect(
                    "Select categorical features",
                    categorical_cols,
                    default=categorical_cols
                )
            else:
                selected_categorical = []
            
            # Combine selections
            if tab2.activated:
                feats = selected_numeric + selected_categorical
        
        if len(feats) == 0:
            st.markdown("""
            <div class="error-message">
                <strong>⚠️ Error:</strong> Please select at least one feature for modeling.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**Selected {len(feats)} features for modeling**")
            
            X = df[feats]
            numeric_feats = X.select_dtypes(include='number').columns.tolist()
            categorical_feats = X.select_dtypes(exclude='number').columns.tolist()
            
            # Add this check:
            if problem == 'Regression' and not numeric_feats:
                st.error("⚠️ Regression models require at least one numeric feature. Please select numeric features.")
                st.stop()  # This halts further execution
            
            # Prepare target
            y = df[target]
            if y.isna().any():
                st.markdown(f"""
                <div class="warning-message">
                    <strong>⚠️ Note:</strong> {y.isna().sum()} missing target values will be dropped.
                </div>
                """, unsafe_allow_html=True)
                valid_idx = y.dropna().index
                X = X.loc[valid_idx]
                y = y.loc[valid_idx]

                if len(X) == 0 or len(y) == 0:
                    st.error("⚠️ Error: No samples remaining after dropping missing target values. Please check your dataset.")
                    st.stop()
            
            # Label encode for classification
            if problem == 'Classification':
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                if min_class_count < 2:
                    st.error(f"""
                    ⚠️ Classification error: Class '{class_counts.idxmin()}' has only {min_class_count} sample(s).
                    Need at least 2 samples per class for stratified split.
                    """)
                    st.markdown("**Solutions:**")
                    st.markdown("1. Collect more data for minority classes")
                    st.markdown("2. Merge similar small classes")
                    st.markdown("3. Switch to non-stratified split")
                    st.stop()

                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                st.session_state.label_encoder = le
                y = pd.Series(y_enc, index=y.index)
                
                # Display target distribution for classification
                st.markdown("#### Target Class Distribution")
                fig, ax = plt.subplots(figsize=(10, 3))
                value_counts = pd.Series(y_enc).value_counts().sort_index()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="viridis")
                ax.set_xlabel(f"{target} (encoded)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
                # Check for class imbalance
                total = len(y)
                min_class_pct = (pd.Series(y_enc).value_counts().min() / total) * 100
                if min_class_pct < 10:
                    st.markdown(f"""
                    <div class="warning-message">
                        <strong>⚠️ Class Imbalance:</strong> The smallest class represents only {min_class_pct:.1f}% of your data.
                        Consider using class weights or balancing techniques.
                    </div>
                    """, unsafe_allow_html=True)
            
            # Train/validation split
            st.markdown("#### Validation Configuration")
            val_pct = st.slider(
                "Validation Set Size (%)", 
                10, 50, 20,
                help="Percentage of data to hold out for validation"
            )
            
            try:
                if problem == 'Classification' and min_class_count >= 2:
                    split_stratify = y
                else:
                    split_stratify = None

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=val_pct/100,
                    stratify=y if problem == 'Classification' else None,
                    random_state=42
                )

                # Check if training or validation set is empty
                if len(X_train) == 0 or len(X_val) == 0:
                    st.error("⚠️ Training or validation set is empty. Please reduce the validation percentage or provide more data.")
                    st.stop()


                st.session_state.X_train = X_train
                st.session_state.y_train = y_train      
                st.session_state.X_val = X_val
                st.session_state.y_val_enc = y_val
                
                # Display split info with progress bars
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: {100-val_pct}%; background-color: #4285F4; color: white; padding: 3px 0; text-align: center; border-radius: 3px 0 0 3px;">
                            Training ({len(X_train)} rows)
                        </div>
                        <div style="width: {val_pct}%; background-color: #34A853; color: white; padding: 3px 0; text-align: center; border-radius: 0 3px 3px 0;">
                            Validation ({len(X_val)} rows)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except ValueError as e:
                st.markdown(f"""
                <div class="error-message">
                    <strong>⚠️ Error splitting data:</strong> {e}
                </div>
                """, unsafe_allow_html=True)
                
            # Model training
            if st.button("🚀 Train Model", type="primary", help="Start the model training process"):
                with st.spinner("Training model... This may take a moment"):
                    # Progress bar animation
                    progress_bar = st.progress(0)
                    for i in range(101):
                        time.sleep(0.01) 
                        progress_bar.progress(i)
                        
                    # Setup preprocessing pipeline
                    num_pipe = Pipeline([
                        ('impute', SimpleImputer(strategy=st.session_state.num_impute)),
                        ('scale', 
                        StandardScaler() if st.session_state.scaling == 'Standardization' else
                        MinMaxScaler() if st.session_state.scaling == 'Normalization' else 'passthrough')
                    ])

                    cat_pipe = Pipeline([
                        ('impute', SimpleImputer(strategy=st.session_state.cat_impute)),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                    ])
                    
                    preproc = ColumnTransformer([
                        ('nums', num_pipe, numeric_feats),
                        ('cats', cat_pipe, categorical_feats)
                    ])
        
                    if model_type == 'Random Forest':
                        Model = RandomForestClassifier if problem == 'Classification' else RandomForestRegressor
                    elif model_type == 'Gradient Boosting':
                        Model = GradientBoostingClassifier if problem == 'Classification' else GradientBoostingRegressor
                    else:
                        Model = SVC if problem == 'Classification' else SVR
                    

                    # pipeline = Pipeline([
                    #     ('pre', preproc),
                    #     ('select', SelectKBest(
                    #         score_func=(f_classif if problem == 'Classification' else mutual_info_regression),
                    #         k='all')),
                    #     ('model', Model())
                    # ])
        
                    # pipeline.fit(st.session_state.X_train, st.session_state.y_train)

                    # Replace the existing pipeline definition with:
                    pipeline_steps = [
                        ('pre', preproc),
                    ]

                    # Only add SelectKBest if there are features to select
                    if len(numeric_feats + categorical_feats) > 0:
                        pipeline_steps.append(
                            ('select', SelectKBest(
                                score_func=(f_classif if problem == 'Classification' else mutual_info_regression),
                                k='all'))
                        )

                    pipeline_steps.append(('model', Model()))
                    pipeline = Pipeline(pipeline_steps)
                    pipeline.fit(st.session_state.X_train, st.session_state.y_train)

                    st.session_state.trained_model = pipeline
                    st.session_state.model_trained = True
                
                st.markdown("""
                <div class="success-message">
                    <strong>✅ Success!</strong> Model trained successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Display feature importances if available
                clf = pipeline.named_steps['model']
                if hasattr(clf, 'feature_importances_'):
                    st.markdown("#### Feature Importances")
                    all_features = pipeline.named_steps['pre'].get_feature_names_out()
                    selected_idx = pipeline.named_steps['select'].get_support(indices=True)
                    feat_names = all_features[selected_idx]
                    importances = clf.feature_importances_
                    
                    # Sort importances
                    indices = np.argsort(importances)[::-1]
                    top_features = feat_names[indices][:15]  # Top 15 features
                    top_importances = importances[indices][:15]
                    
                    # Plot feature importances with better styling
                    fig, ax = plt.subplots(figsize=(10, min(8, len(top_features)/2+1)))
                    plt.barh(range(len(top_importances)), top_importances, align='center', color='#4285F4')
                    plt.yticks(range(len(top_importances)), top_features)
                    plt.gca().invert_yaxis()
                    plt.xlabel('Importance')
                    plt.title('Top Feature Importances')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Next step button
        if st.session_state.get('model_trained'):
            logger.info("Model Trained")
            if st.button("Continue to Evaluation →"):
                logger.info(f"BEFORE: session_state.current_step: {st.session_state.current_step}")
                st.session_state.current_step = 4
                logger.info(f"AFTER: session_state.current_step: {st.session_state.current_step}")
                st.rerun()
        
        logger.info(f"Step-3: session_state.current_step: {st.session_state.current_step}")
        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            logger.info("Entering Col1")
            logger.info(f"Col1: session_state.current_step: {st.session_state.current_step}")
            if st.button("← Go Back", key="model_back"):
                st.session_state.current_step = 2
                st.rerun()

# MODEL EVALUATION (STEP 4)
# elif current_step == 4 or (nav_option == "Evaluation" and current_step != 4):
elif st.session_state.current_step == 4:
    # st.session_state.current_step = 4
    
    st.markdown("""
    <h2><i class="fas fa-chart-bar"></i> Model Evaluation</h2>
    <p>Analyze your model's performance with detailed metrics and visualizations</p>
    """, unsafe_allow_html=True)
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ No trained model found. Please complete the model training step first.")
        if st.button("← Go to Model Training"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        pipe = st.session_state.trained_model
        Xv = st.session_state.X_val
        yv = st.session_state.y_val_enc
        
        problem = st.session_state.problem
        
        with st.spinner("Generating evaluation report..."):
            preds = pipe.predict(Xv)
            
            st.markdown("### Performance Metrics")
            
            if problem == 'Classification':
                # Process validation set for SHAP
                Xv_proc = pipe.named_steps['pre'].transform(Xv)
                if hasattr(Xv_proc, "toarray"):
                    Xv_proc = Xv_proc.toarray()
                Xv_proc = Xv_proc.astype(np.float64)
                feature_names = pipe.named_steps['pre'].get_feature_names_out()
                
                # Core metrics
                acc = accuracy_score(yv, preds)
                f1 = f1_score(yv, preds, average='macro')
                
                # Display in nice cards
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <p class="metric-label">Accuracy</p>
                        <p class="metric-value">{:.1f}%</p>
                    </div>
                    """.format(acc*100), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <p class="metric-label">F1 Score (Macro)</p>
                        <p class="metric-value">{:.1f}%</p>
                    </div>
                    """.format(f1*100), unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                eval_tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
                
                with eval_tabs[0]:
                    # Confusion Matrix
                    if hasattr(st.session_state, "label_encoder") and st.session_state.label_encoder is not None:
                        le = st.session_state.label_encoder
                        class_names = le.classes_
                    else:
                        class_names = np.unique(yv)
                    
                    cm = pd.crosstab(yv, preds, rownames=['Actual'], colnames=['Predicted'])
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                                xticklabels=class_names, yticklabels=class_names)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate and show class-wise metrics
                    st.markdown("##### Class-wise Performance")
                    class_metrics = pd.DataFrame({
                        'Class': class_names,
                        'Precision': [precision_score(yv, preds, average=None, labels=[i])[0] for i in range(len(class_names))],
                        'Recall': [recall_score(yv, preds, average=None, labels=[i])[0] for i in range(len(class_names))], 
                        'F1 Score': [f1_score(yv, preds, average=None, labels=[i])[0] for i in range(len(class_names))]
                    })
                    st.dataframe(class_metrics.style.format({
                        'Precision': '{:.1%}',
                        'Recall': '{:.1%}',
                        'F1 Score': '{:.1%}'
                    }))
                
                with eval_tabs[1]:
                    # ROC Curve
                    if hasattr(pipe, 'predict_proba'):
                        # For multi-class, we'll show ROC for each class (one-vs-rest)
                        proba = pipe.predict_proba(Xv)
                        
                        n_classes = proba.shape[1]
                        
                        # Create ROC curve for each class
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(yv == i, proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            
                            class_label = class_names[i] if i < len(class_names) else f"Class {i}"
                            ax.plot(fpr, tpr, lw=2, 
                                   label=f'{class_label} (AUC = {roc_auc:.2f})')
                        
                        ax.plot([0, 1], [0, 1], 'k--', lw=2)
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curves (One-vs-Rest)')
                        ax.legend(loc="lower right")
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    else:
                        st.info("This model does not support probability predictions required for ROC curves.")
                
                with eval_tabs[2]:
                    # SHAP values for feature importance
                    try:
                        st.markdown("#### SHAP Feature Importance")
                        st.info("Calculating SHAP values... This may take a moment for complex models.")
                        
                        # Only compute SHAP for a sample if the dataset is large
                        sample_size = min(100, Xv_proc.shape[0])
                        sample_indices = np.random.choice(Xv_proc.shape[0], sample_size, replace=False)
                        
                        model = pipe.named_steps['model']
                        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                        sv = explainer.shap_values(Xv_proc[sample_indices], check_additivity=False)
                        
                        if isinstance(sv, list):  # For multi-class models
                            # Show summary for the first class
                            plt.figure()
                            shap.summary_plot(sv[0], Xv_proc[sample_indices], feature_names=feature_names, 
                                             plot_type="bar", show=False)
                            plt.title(f"SHAP Feature Importance (Class: {class_names[0]})")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
                            # Show detailed plot
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(sv[0], Xv_proc[sample_indices], feature_names=feature_names, show=False)
                            plt.title(f"SHAP Summary Plot (Class: {class_names[0]})")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                        else:
                            plt.figure()
                            shap.summary_plot(sv, Xv_proc[sample_indices], feature_names=feature_names, 
                                             plot_type="bar", show=False)
                            plt.title("SHAP Feature Importance")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
                            plt.figure(figsize=(10, 8))
                            shap.summary_plot(sv, Xv_proc[sample_indices], feature_names=feature_names, show=False)
                            plt.title("SHAP Summary Plot")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                    except Exception as e:
                        st.error(f"Error calculating SHAP values: {e}")
                        st.info("SHAP analysis is not available for this model configuration.")
                
            else:  # Regression metrics
                # Core metrics
                mse = mean_squared_error(yv, preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(yv, preds)
                mae = mean_absolute_error(yv, preds)
                
                # Display metrics in 3 nice cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <p class="metric-label">R² Score</p>
                        <p class="metric-value">{:.3f}</p>
                    </div>
                    """.format(r2), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <p class="metric-label">RMSE</p>
                        <p class="metric-value">{:.3f}</p>
                    </div>
                    """.format(rmse), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <p class="metric-label">MAE</p>
                        <p class="metric-value">{:.3f}</p>
                    </div>
                    """.format(mae), unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                eval_tabs = st.tabs(["Prediction Plot", "Residuals", "Feature Importance"])
                
                with eval_tabs[0]:
                    # Actual vs Predicted
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(yv, preds, alpha=0.6, edgecolor='k', s=50)
                    ax.plot([yv.min(), yv.max()], [yv.min(), yv.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs Predicted Values')
                    # Add R² annotation
                    ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with eval_tabs[1]:
                    # Residual plot
                    residuals = yv - preds
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Residuals vs Predicted
                    ax1.scatter(preds, residuals, alpha=0.6, edgecolor='k', s=50)
                    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax1.set_xlabel('Predicted Values')
                    ax1.set_ylabel('Residuals')
                    ax1.set_title('Residuals vs Predicted Values')
                    ax1.grid(True, alpha=0.3)
                    
                    # Residual distribution
                    sns.histplot(residuals, kde=True, ax=ax2)
                    ax2.set_xlabel('Residual Value')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Residual Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Additional residual statistics
                    st.markdown("##### Residual Statistics")
                    res_stats = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                        'Value': [
                            residuals.mean(),
                            residuals.std(),
                            residuals.min(),
                            np.percentile(residuals, 25),
                            np.median(residuals),
                            np.percentile(residuals, 75),
                            residuals.max()
                        ]
                    })
                    st.dataframe(res_stats.style.format({'Value': '{:.3f}'}))
                
                with eval_tabs[2]:
                    # Feature importance
                    try:
                        # SHAP for feature importance
                        st.markdown("#### SHAP Feature Importance")
                        st.info("Calculating SHAP values... This may take a moment for complex models.")
                        
                        # Process validation set for SHAP
                        Xv_proc = pipe.named_steps['pre'].transform(Xv)
                        if hasattr(Xv_proc, "toarray"):
                            Xv_proc = Xv_proc.toarray()
                        Xv_proc = Xv_proc.astype(np.float64)
                        feature_names = pipe.named_steps['pre'].get_feature_names_out()
                        
                        # Only compute SHAP for a sample if the dataset is large
                        sample_size = min(100, Xv_proc.shape[0])
                        sample_indices = np.random.choice(Xv_proc.shape[0], sample_size, replace=False)
                        
                        model = pipe.named_steps['model']
                        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                        sv = explainer.shap_values(Xv_proc[sample_indices], check_additivity=False)
                        
                        plt.figure()
                        shap.summary_plot(sv, Xv_proc[sample_indices], feature_names=feature_names, 
                                         plot_type="bar", show=False)
                        plt.title("SHAP Feature Importance")
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(sv, Xv_proc[sample_indices], feature_names=feature_names, show=False)
                        plt.title("SHAP Summary Plot")
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.error(f"Error calculating SHAP values: {e}")
                        st.info("SHAP analysis is not available for this model configuration.")
        
        # Next/Previous buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Go Back", key="eval_back"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("Continue to Deployment →", type="primary"):
                st.session_state.current_step = 5
                st.rerun()

# MODEL DEPLOYMENT (STEP 5)
# elif current_step == 5 or (nav_option == "Deployment" and current_step != 5):
elif st.session_state.current_step == 5:
    # st.session_state.current_step = 5
    
    st.markdown("""
    <h2><i class="fas fa-rocket"></i> Model Deployment</h2>
    <p>Export your model or make batch predictions on new data</p>
    """, unsafe_allow_html=True)
    
    if st.session_state.trained_model is None:
        st.warning("⚠️ No trained model found. Please complete the model training step first.")
        if st.button("← Go to Model Training"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                <div class="section-container" style="height: 100%;">
                    <h3 style="margin-top: 0;">Export Model</h3>
                    <p>Download your trained model to use in other applications</p>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <div style="font-size: 4rem; margin-bottom: 20px;">💾</div>
                        <p>Your model is ready for export</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("📥 Download Model", type="primary"):
                joblib.dump(st.session_state.trained_model, "trained_model.pkl")
                with open("trained_model.pkl", "rb") as f:
                    st.download_button(
                        "⬇️ Download PKL File", 
                        f, 
                        file_name="trained_model.pkl",
                        mime="application/octet-stream"
                    )
            
            st.markdown("""
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-container" style="height: 100%;">
                <h3 style="margin-top: 0;">Batch Prediction</h3>
                <p>Make predictions on new data with your trained model</p>
            """, unsafe_allow_html=True)
            
            pred_file = st.file_uploader("Upload CSV with new data for prediction", key="pred")
            if pred_file:
                try:
                    with st.spinner("Processing prediction data..."):
                        df2 = pd.read_csv(pred_file)
                        
                        # Check if the required columns exist
                        model_features = st.session_state.X_val.columns.tolist()
                        missing_cols = [col for col in model_features if col not in df2.columns]
                        
                        if missing_cols:
                            st.markdown(f"""
                            <div class="error-message">
                                <strong>⚠️ Error:</strong> Missing required columns: {', '.join(missing_cols)}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Make predictions
                            preds2 = st.session_state.trained_model.predict(df2[model_features])
                            
                            # Convert back to original labels for classification
                            if st.session_state.problem == 'Classification' and st.session_state.label_encoder is not None:
                                le = st.session_state.label_encoder
                                preds2 = le.inverse_transform(preds2)
                            
                            # Add predictions to the dataframe
                            df2['prediction'] = preds2
                            
                            # Show predictions
                            st.markdown("#### Prediction Results")
                            st.dataframe(df2, use_container_width=True)
                            
                            # Download predictions
                            csv = df2.to_csv(index=False).encode()
                            st.download_button(
                                "⬇️ Download Predictions", 
                                csv, 
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Visualization for predictions
                            if st.session_state.problem == 'Classification':
                                st.markdown("#### Prediction Distribution")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                pred_counts = pd.Series(preds2).value_counts().sort_index()
                                sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax, palette="viridis")
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                            else:
                                st.markdown("#### Prediction Distribution")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                sns.histplot(preds2, kde=True, ax=ax)
                                ax.set_xlabel("Predicted Value")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error processing predictions: {e}")
            
            st.markdown("""
            </div>
            """, unsafe_allow_html=True)
        
        # Go back button
        if st.button("← Back to Evaluation"):
            st.session_state.current_step = 4
            st.rerun()

# Close the section container div
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 3rem; text-align: center; color: #888;">
    <p>ML Pipeline Builder • Designed with ❤️ • &copy; 2025</p>
</div>
""", unsafe_allow_html=True)
