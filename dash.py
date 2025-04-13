import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import re
from io import StringIO

# Load the Key API from the environment
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
)



# Page configuration
st.set_page_config(
    page_title="LLM4Dash | AI-Powered Vis Dashboard",
    page_icon="🤖📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
    using_secrets = True
except:
    client = None
    api_key = None
    using_secrets = False

# Title and description
st.title("🤖📊 LLM4Dash | AI-Powered Visualization Dashboard")
st.markdown("""
Upload your data and LLM4Dash will analyze its structure and recommend the best visualizations.
""")

# Sidebar for API configuration
with st.sidebar:    
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, Excel)", 
        type=["csv", "xlsx", "xls"]
    )
    use_sample_data = st.checkbox("Use sample data", value=True)

# Function to analyze data with LLM
def analyze_data_with_llm(df, sample_size=5):
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    
    # Create data sample for the prompt
    sample_data = df.head(sample_size).to_string()
    
    prompt = f"""
    Analyze the following dataset and provide:
    1. Classification of each column (nominal qualitative, ordinal qualitative, discrete quantitative, continuous quantitative, date/time, geolocation)
    2. Recommendations for suitable chart types for these variables
    3. Interesting variable combinations to visualize

    Expected response format:
    **Column Analysis:**
    - [Column name]: [Data type], [Brief explanation]
    
    **Visualization Recommendations:**
    - [Recommended chart type]: [Suggested variables] - [Reason]
    
    **Interesting Combinations:**
    - [Variable 1] vs [Variable 2]: [Chart type] - [Reason]

    Data:
    {sample_data}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error connecting to OpenAI API: {e}")
        return None

# Function to extract recommendations from analysis
def parse_llm_response(response):
    if not response:
        return None
    
    sections = {
        "Column Analysis": "",
        "Visualization Recommendations": "",
        "Interesting Combinations": ""
    }
    
    current_section = None
    for line in response.split('\n'):
        if line.startswith("**Column Analysis:**"):
            current_section = "Column Analysis"
        elif line.startswith("**Visualization Recommendations:**"):
            current_section = "Visualization Recommendations"
        elif line.startswith("**Interesting Combinations:**"):
            current_section = "Interesting Combinations"
        elif current_section and line.strip() and not line.startswith("**"):
            sections[current_section] += line + "\n"
    
    return sections

# Load sample data
@st.cache_data
def load_sample_data():
    iris = sns.load_dataset('iris')
    tips = sns.load_dataset('tips')
    gapminder = px.data.gapminder()
    stocks = px.data.stocks()
    
    return {
        "Iris Dataset": iris,
        "Tips Dataset": tips,
        "Gapminder Dataset": gapminder,
        "Stocks Dataset": stocks
    }

# Load data
if use_sample_data or uploaded_file is None:
    sample_data = load_sample_data()
    selected_dataset = st.selectbox("Select sample dataset", list(sample_data.keys()))
    df = sample_data[selected_dataset]
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

# Display data
with st.expander("🔍 View Data"):
    st.dataframe(df.head(50))
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("**Descriptive statistics**")
    st.dataframe(df.describe(include='all'))


# Automatic analysis with LLM


# ... [previous code remains the same until the Automatic analysis with LLM section]

# Automatic analysis with LLM
# ... [previous code remains the same until the analysis section]

# Automatic analysis with LLM
if st.button("🔍 Analyze Data with AI") and api_key:
    with st.spinner("Analyzing data with AI..."):
        analysis = analyze_data_with_llm(df)
        
        if analysis:
            parsed_analysis = parse_llm_response(analysis)
            
            with st.expander("📝 Automatic Data Analysis", expanded=True):
                if parsed_analysis["Column Analysis"]:
                    st.subheader("Column Classification")
                    st.markdown(parsed_analysis["Column Analysis"])
                
                if parsed_analysis["Visualization Recommendations"]:
                    st.subheader("Visualization Recommendations")
                
                    st.markdown("**OpenAI's Recommendations Text**")
                    st.markdown(
                        f"""
                        {parsed_analysis["Visualization Recommendations"]}
                        """,
                        unsafe_allow_html=True
                    )
                   
                if parsed_analysis["Interesting Combinations"]:
                    st.subheader("Interesting Combinations")
                    st.markdown(parsed_analysis["Interesting Combinations"])
else:
    if not api_key:
        st.warning("Enter an OpenAI API key to use automatic analysis")

# ... [rest of the code remains the same]

# Automatic column classification (LLM fallback)
def classify_columns(df):
    column_types = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        # Date detection
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "Date/Time"
            continue
            
        # Try to convert to date
        try:
            pd.to_datetime(df[col])
            column_types[col] = "Date/Time"
            continue
        except:
            pass
            
        # Geolocation detection (place names or coordinates)
        if col.lower() in ['country', 'city', 'state', 'region', 'address', 'location']:
            column_types[col] = "Geolocation (nominal)"
            continue
            
        # Coordinate detection
        if col.lower() in ['lat', 'latitude', 'lon', 'longitude']:
            column_types[col] = "Geolocation (coordinate)"
            continue
            
        # Numeric variables
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = len(col_data.unique())
            
            if unique_vals < 10 and all(val.is_integer() for val in col_data if not pd.isna(val)):
                column_types[col] = "Discrete quantitative (low cardinality)"
            elif unique_vals < 20:
                column_types[col] = "Discrete quantitative"
            else:
                column_types[col] = "Continuous quantitative"
            continue
            
        # Categorical variables
        unique_vals = len(col_data.unique())
        if unique_vals < 10:
            column_types[col] = "Nominal qualitative" if not pd.api.types.is_numeric_dtype(df[col]) else "Ordinal qualitative"
        else:
            column_types[col] = "Text" if unique_vals/len(col_data) > 0.5 else "Nominal qualitative (high cardinality)"
    
    return column_types


# Generate recommendations based on data types
def generate_recommendations(column_types):
    recommendations = []
    numeric_vars = [col for col, typ in column_types.items() if "quantitative" in typ]
    cat_vars = [col for col, typ in column_types.items() if "qualitative" in typ]
    date_vars = [col for col, typ in column_types.items() if "Date" in typ]
    geo_vars = [col for col, typ in column_types.items() if "Geolocation" in typ]
    
    # Recommendations for numeric variables
    if len(numeric_vars) >= 1:
        recommendations.append(f"- **Histogram**: For {numeric_vars[0]} - Shows value distribution")
        
    if len(numeric_vars) >= 2:
        recommendations.append(f"- **Scatter plot**: {numeric_vars[0]} vs {numeric_vars[1]} - Relationship between two numeric variables")
    
    # Recommendations for categorical variables
    if len(cat_vars) >= 1 and len(numeric_vars) >= 1:
        recommendations.append(f"- **Box plot**: {cat_vars[0]} vs {numeric_vars[0]} - Distribution of numeric values by category")
        recommendations.append(f"- **Bar plot**: Value counts for {cat_vars[0]}")
    
    # Recommendations for dates
    if len(date_vars) >= 1 and len(numeric_vars) >= 1:
        recommendations.append(f"- **Line chart**: {date_vars[0]} vs {numeric_vars[0]} - Time series")
    
    # Recommendations for geolocation
    if len(geo_vars) >= 1:
        if any("coordinate" in column_types[col] for col in geo_vars):
            lat_col = next((col for col in geo_vars if "lat" in col.lower()), None)
            lon_col = next((col for col in geo_vars if "lon" in col.lower()), None)
            if lat_col and lon_col:
                recommendations.append(f"- **Scatter map**: {lat_col} vs {lon_col} - Geographic visualization")
        
        if any("nominal" in column_types[col] for col in geo_vars):
            geo_col = next((col for col in geo_vars if "nominal" in column_types[col]), None)
            if geo_col and len(numeric_vars) >= 1:
                recommendations.append(f"- **Choropleth map**: {geo_col} colored by {numeric_vars[0]}")
    
    return recommendations


# Visualization configuration
st.header("📈 Configure Visualization following the AI recommendations")

# Chart type selection based on recommendations
chart_types = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
    "Box Plot", "Violin Plot", "Heatmap", "Pie Chart", 
    "Area Chart", "3D Scatter", "Map"
]

selected_chart = st.selectbox("Chart type", chart_types)

# Generate chart based on selection (similar to previous code)
if selected_chart == "Scatter Plot":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    fig = px.scatter(df, x=x_var, y=y_var, title=f"Scatter Plot: {x_var} vs {y_var}")
    st.plotly_chart(fig)
elif selected_chart == "Line Chart":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    fig = px.line(df, x=x_var, y=y_var, title=f"Line Chart: {x_var} vs {y_var}")
    st.plotly_chart(fig)
elif selected_chart == "Bar Chart":
    x_var = st.selectbox("X-axis variable", df.columns)
    fig = px.bar(df[x_var].value_counts().reset_index(), 
                 x='index', y=x_var, 
                 title=f"Bar Chart: {x_var} Distribution")
    st.plotly_chart(fig)
elif selected_chart == "Histogram":
    x_var = st.selectbox("Variable", df.columns)
    fig = px.histogram(df, x=x_var, title=f"Histogram of {x_var}")
    st.plotly_chart(fig)
elif selected_chart == "Box Plot":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    fig = px.box(df, x=x_var, y=y_var, title=f"Box Plot: {x_var} vs {y_var}")
    st.plotly_chart(fig)
elif selected_chart == "Violin Plot":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    fig = px.violin(df, x=x_var, y=y_var, title=f"Violin Plot: {x_var} vs {y_var}")
    st.plotly_chart(fig)
elif selected_chart == "Heatmap":
    corr = df.corr()
    fig = px.imshow(corr, title="Correlation Heatmap")
    st.plotly_chart(fig)
elif selected_chart == "Pie Chart":
    x_var = st.selectbox("Variable", df.columns)
    fig = px.pie(df, names=x_var, title=f"Pie Chart of {x_var}")
    st.plotly_chart(fig)
elif selected_chart == "Area Chart":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    fig = px.area(df, x=x_var, y=y_var, title=f"Area Chart: {x_var} vs {y_var}")
    st.plotly_chart(fig)
elif selected_chart == "3D Scatter":
    x_var = st.selectbox("X-axis variable", df.columns)
    y_var = st.selectbox("Y-axis variable", df.columns)
    z_var = st.selectbox("Z-axis variable", df.columns)
    fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, 
                        title=f"3D Scatter Plot: {x_var} vs {y_var} vs {z_var}")
    st.plotly_chart(fig)
elif selected_chart == "Map":
    lat_col = st.selectbox("Latitude variable", df.columns)
    lon_col = st.selectbox("Longitude variable", df.columns)
    if lat_col and lon_col:
        fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, 
                                mapbox_style="carto-positron", 
                                zoom=3, title="Map Visualization")
        st.plotly_chart(fig)
    else:
        st.warning("Select latitude and longitude variables for map visualization")
elif selected_chart == "Custom":
    st.warning("Custom visualizations are not yet implemented. Please select a predefined chart type.")
else:
    st.warning("Select a chart type to visualize your data.")
# Download options
st.header("📥 Download Chart")

st.download_button(
    label="Download Chart as PNG",
    data=fig.to_image(format='png'),
    file_name='chart.png',
    mime='image/png'
)


# Footer
st.markdown("---")
st.markdown("🤖📊 **AI-Powered Visualization Dashboard** | Created By Dario Ceballos | Built with Streamlit, Plotly and OpenAI")