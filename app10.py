import streamlit as st

# Set page configuration (must be the very first Streamlit command)
st.set_page_config(
    page_title="Advanced Crop Yield Analysis & Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import datetime
import io
import base64

# Scikit-learn imports for modeling
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --------------------------------
# Custom CSS for Dashboard Feel
# --------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .css-1d391kg { 
        background-color: #f1f5f9;
    }
    .stButton>button {
        background-color: #ff9933;
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .highlight {
        color: #047857;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data Explorer", "Yield Prediction"])

# Display last updated time in sidebar
current_time = datetime.datetime.now().strftime("%B %d, %Y %H:%M")
st.sidebar.markdown(f"<p style='text-align: center; color: gray;'>Last updated: {current_time}</p>", unsafe_allow_html=True)

# --------------------------------
# Data Loading & Feature Engineering
# --------------------------------
@st.cache(allow_output_mutation=True)
def load_data():
    try:
        df = pd.read_csv("cleaned_crop_yield.csv")
        # Derived features
        df['Fertilizer_per_area'] = df['Fertilizer'] / df['Area']
        df['Pesticide_per_area'] = df['Pesticide'] / df['Area']
        df['Production_per_area'] = df['Production'] / df['Area']
        # Year-over-year changes can be computed if sorted properly:
        df = df.sort_values(['State', 'Crop', 'Season', 'Crop_Year'])
        df['Previous_Year_Yield'] = df.groupby(['State', 'Crop', 'Season'])['Yield'].shift(1)
        df['YoY_Yield_Change'] = df['Yield'] - df['Previous_Year_Yield']
        df['YoY_Yield_Change_Pct'] = (df['YoY_Yield_Change'] / df['Previous_Year_Yield']) * 100
        return df
    except Exception as e:
        st.error("Error loading data: Ensure 'cleaned_crop_yield.csv' is in the correct location.")
        return pd.DataFrame()

df = load_data()

# Define common filter options (if data is available)
if not df.empty:
    crop_options = ['All'] + sorted(df['Crop'].unique().tolist())
    state_options = ['All'] + sorted(df['State'].unique().tolist())
    season_options = ['All'] + sorted(df['Season'].unique().tolist())
    min_year = int(df['Crop_Year'].min())
    max_year = int(df['Crop_Year'].max())
else:
    crop_options = state_options = season_options = []
    min_year, max_year = 2000, 2020

# --------------------------------
# Natural Language Query Processing Function
# --------------------------------
def process_query(query_text, crop_opts, state_opts, season_opts, yr_range):
    query_lower = query_text.lower()
    crop = "All"
    for opt in crop_opts:
        if opt != "All" and opt.lower() in query_lower:
            crop = opt
            break
    state = "All"
    for opt in state_opts:
        if opt != "All" and opt.lower() in query_lower:
            state = opt
            break
    season = "All"
    for opt in season_opts:
        if opt != "All" and opt.lower() in query_lower:
            season = opt
            break
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', query_text)
    if len(years) >= 2:
        yr_range_query = (int(min(years)), int(max(years)))
    elif len(years) == 1:
        yr_range_query = (int(years[0]), int(years[0]))
    else:
        yr_range_query = yr_range
    return crop, state, season, yr_range_query

# --------------------------------
# Model Training: Stacking Ensemble with Two Base Models
# --------------------------------
@st.cache(allow_output_mutation=True)
def train_stacking_model(data):
    # Features and target
    features = ['Crop', 'Crop_Year', 'Season', 'State', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Area']
    target = 'Yield'
    X = data[features]
    y = data[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numerical features
    categorical_features = ['Crop', 'Season', 'State']
    numerical_features = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Area']
    
    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Define base models
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ]
    
    # Meta learner
    meta_learner = LinearRegression()
    
    # Create Stacking Regressor
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacking', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict and compute performance metrics on test set
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    metrics = {"R2 Score": r2, "RMSE": rmse, "MAE": mae}
    
    return pipeline, metrics

stack_model, model_metrics = train_stacking_model(df)

# --------------------------------
# Page 1: Dashboard
# --------------------------------
if page == "Dashboard":
    st.title("Crop Yield Dashboard")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Crops", len(df["Crop"].unique()))
    with col2:
        st.metric("States Covered", len(df["State"].unique()))
    with col3:
        st.metric("Average Yield", f"{df['Yield'].mean():.2f}")
    
    st.markdown("---")
    
    # National Yield Trend
    st.subheader("National Yield Trend")
    yearly = df.groupby("Crop_Year")["Yield"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly["Crop_Year"], yearly["Yield"], marker="o", color="#1e3a8a")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Yield")
    ax.set_title("National Average Yield Over Years")
    ax.grid(True)
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    ax2.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig2)
    
    # Interactive Plotly Scatter (Fertilizer per Area vs. Yield)
    st.subheader("Interactive Scatter Plot")
    fig3 = px.scatter(df, x="Fertilizer_per_area", y="Yield", color="State", 
                      title="Yield vs Fertilizer per Area by State",
                      labels={"Fertilizer_per_area": "Fertilizer per Area", "Yield": "Crop Yield"})
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------
# Page 2: Data Explorer
# --------------------------------
elif page == "Data Explorer":
    st.title("Data Explorer")
    
    # Sidebar Filters
    st.sidebar.subheader("Filter Options")
    sel_crop = st.sidebar.selectbox("Select Crop", crop_options)
    sel_state = st.sidebar.selectbox("Select State", state_options)
    sel_season = st.sidebar.selectbox("Select Season", season_options)
    yr_range = st.sidebar.slider("Select Crop Year Range", min_year, max_year, (min_year, max_year))
    
    st.sidebar.subheader("Natural Language Query")
    query_input = st.sidebar.text_input("E.g., 'Show me wheat in Punjab between 2010 and 2018'")
    
    if query_input:
        crop_filter, state_filter, season_filter, yr_range_filter = process_query(query_input, crop_options, state_options, season_options, yr_range)
        st.sidebar.markdown("**Interpreted Filters:**")
        st.sidebar.write(f"Crop: {crop_filter}")
        st.sidebar.write(f"State: {state_filter}")
        st.sidebar.write(f"Season: {season_filter}")
        st.sidebar.write(f"Year Range: {yr_range_filter}")
    else:
        crop_filter, state_filter, season_filter, yr_range_filter = sel_crop, sel_state, sel_season, yr_range

    # Data Filtering
    filtered_df = df.copy()
    if crop_filter != "All":
        filtered_df = filtered_df[filtered_df["Crop"] == crop_filter]
    if state_filter != "All":
        filtered_df = filtered_df[filtered_df["State"] == state_filter]
    if season_filter != "All":
        filtered_df = filtered_df[filtered_df["Season"] == season_filter]
    filtered_df = filtered_df[(filtered_df["Crop_Year"] >= yr_range_filter[0]) & (filtered_df["Crop_Year"] <= yr_range_filter[1])]
    
    st.subheader("Filtered Data")
    st.write(f"{len(filtered_df)} records found.")
    st.dataframe(filtered_df)
    
    # Visualization: Yield Trend by State
    st.subheader("Yield Trend by State")
    if not filtered_df.empty:
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        for state in filtered_df["State"].unique():
            state_data = filtered_df[filtered_df["State"] == state].groupby("Crop_Year")["Yield"].mean().reset_index()
            ax4.plot(state_data["Crop_Year"], state_data["Yield"], marker="o", label=state)
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Average Yield")
        ax4.set_title("Yield Trend by State")
        ax4.grid(True)
        ax4.legend(loc="upper left", bbox_to_anchor=(1,1))
        st.pyplot(fig4)
    
    # Visualization: Yield Comparison by Crop
    st.subheader("Yield Comparison by Crop")
    if not filtered_df.empty:
        crop_avg = filtered_df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).reset_index()
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        ax5.bar(crop_avg["Crop"], crop_avg["Yield"], color="#ff9933")
        ax5.set_xlabel("Crop")
        ax5.set_ylabel("Average Yield")
        ax5.set_title("Average Yield by Crop")
        plt.xticks(rotation=45)
        st.pyplot(fig5)
    
    # Download link for filtered data
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="filtered_crop_data.csv">Download Filtered Data as CSV</a>', unsafe_allow_html=True)

# --------------------------------
# Page 3: Yield Prediction with Stacking Ensemble
# --------------------------------
elif page == "Yield Prediction":
    st.title("Crop Yield Prediction")
    st.write("Enter details below to predict the crop yield and view model accuracy metrics.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            pred_crop = st.selectbox("Crop", sorted(df["Crop"].unique().tolist()))
            pred_year = st.number_input("Crop Year", min_value=min_year, max_value=max_year, value=min_year)
            pred_season = st.selectbox("Season", sorted(df["Season"].unique().tolist()))
        with col2:
            pred_state = st.selectbox("State", sorted(df["State"].unique().tolist()))
            pred_rainfall = st.number_input("Annual Rainfall", value=float(df["Annual_Rainfall"].mean()))
            pred_area = st.number_input("Area", value=float(df["Area"].mean()))
            pred_fertilizer = st.number_input("Fertilizer", value=float(df["Fertilizer"].mean()))
            pred_pesticide = st.number_input("Pesticide", value=float(df["Pesticide"].mean()))
        submitted = st.form_submit_button("Predict Yield")
    
    if submitted:
        # Prepare input data
        input_df = pd.DataFrame({
            "Crop": [pred_crop],
            "Crop_Year": [pred_year],
            "Season": [pred_season],
            "State": [pred_state],
            "Annual_Rainfall": [pred_rainfall],
            "Fertilizer": [pred_fertilizer],
            "Pesticide": [pred_pesticide],
            "Area": [pred_area]
        })
        prediction = stack_model.predict(input_df)[0]
        st.success(f"Predicted Crop Yield: {prediction:.2f}")
        
        # Display model accuracy metrics
        st.write("### Model Performance on Test Set")
        st.write(model_metrics)
