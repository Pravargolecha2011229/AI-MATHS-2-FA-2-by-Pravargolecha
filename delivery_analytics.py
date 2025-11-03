import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Any
import logging
from datetime import datetime
import json
import hashlib
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state for user management
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'history' not in st.session_state:
    st.session_state.history = {}

def load_users() -> dict:
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users: dict):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def login(username: str, password: str) -> bool:
    users = load_users()
    if username in users and users[username]['password'] == hash_password(password):
        st.session_state.user_logged_in = True
        st.session_state.username = username
        if username not in st.session_state.history:
            st.session_state.history[username] = []
        return True
    return False

def signup(username: str, password: str) -> bool:
    users = load_users()
    if username in users:
        return False
    users[username] = {
        'password': hash_password(password),
        'history': []
    }
    save_users(users)
    return True

def add_to_history(action: str):
    if st.session_state.username:
        st.session_state.history.setdefault(st.session_state.username, []).append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'action': action
        })

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration with modern settings
st.set_page_config(
    page_title="Delivery Analytics Platform",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Advanced Delivery Analytics Platform\nVersion 2.1"
    }
)

# Enhanced Modern CSS with dark mode support
st.markdown("""
    <style>
    /* Modern Color Scheme */
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --background-color: #FFFFFF;
        --text-color: #333333;
        --metric-bg: #f8f9fa;
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #1E1E1E;
            --text-color: #E0E0E0;
            --metric-bg: #2D2D2D;
        }
    }

    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1rem 2rem;
    }

    /* Modern Card Design */
    .stMetric {
        background-color: var(--metric-bg);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
    }

    .stMetric label {
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Modern Typography */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1.5rem 0;
    }

    /* Enhanced Sidebar */
    .css-1d391kg {
        background-color: var(--metric-bg);
    }

    /* Modern Button Styling */
    .stButton>button {
        border-radius: 25px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Loading Animation */
    .stSpinner {
        border-width: 4px;
    }

    /* Progress Bar Enhancement */
    .stProgress > div > div {
        background-color: var(--primary-color);
        transition: width 0.3s ease;
    }
    /* Sidebar polish - dark theme */
    .sidebar .css-1d391kg, .stSidebar, .css-1d391kg {
        background: #000000 !important;
        color: #FFFFFF !important;
        padding: 1rem 0.7rem;
        border-radius: 0 12px 12px 0;
    }
    /* Ensure all sidebar text and links are white for contrast */
    .sidebar .css-1d391kg * , .stSidebar * , .css-1d391kg * {
        color: #FFFFFF !important;
    }
    .sidebar .css-1d391kg a, .stSidebar a, .css-1d391kg a {
        color: #FFFFFF !important;
    }
    /* Style sidebar buttons for dark background */
    .sidebar .stButton>button, .stSidebar .stButton>button {
        background: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    /* Carded tab area */
    .carded {
        background-color: var(--metric-bg);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Download button styling */
    button[title] {
        border-radius: 12px !important;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== DATA LOADING & CLEANING ====================

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, tuple, tuple, float]:
    """
    Load and clean the delivery dataset with comprehensive preprocessing.
    
    Args:
        file_path (str): Path to the Excel data file
        
    Returns:
        Tuple[pd.DataFrame, tuple, tuple, float]: Cleaned dataframe, original shape, 
        cleaned shape, and late threshold
    """
    try:
        # Construct proper file path using pathlib
        data_path = Path(file_path)
        if not data_path.exists():
            # Try looking in the current directory
            current_dir = Path(__file__).parent
            data_path = current_dir / "Last mile Delivery Data.xlsx"
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {file_path} or current directory")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_excel(data_path, engine='openpyxl')
        
        original_shape = df.shape
        logger.info(f"Original dataset shape: {original_shape}")
        
        # Enhanced data cleaning with Python 3.11 features
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Column mapping for flexible detection
        column_mapping = {
            'delivery_time': ['Delivery_Time', 'delivery_time', 'Time', 'DeliveryTime'],
            'traffic': ['Traffic', 'traffic', 'Traffic_Condition'],
            'weather': ['Weather', 'weather', 'Weather_Condition'],
            'vehicle': ['Vehicle', 'vehicle', 'Vehicle_Type'],
            'agent_age': ['Agent_Age', 'agent_age', 'Age'],
            'agent_rating': ['Agent_Rating', 'agent_rating', 'Rating'],
            'area': ['Area', 'area', 'Location', 'Region'],
            'category': ['Category', 'category', 'Product_Category']
        }
        
        # Flexible column detection
        for standard_name, variations in column_mapping.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # Ensure required columns exist
        required_cols = ['delivery_time', 'traffic', 'weather', 'vehicle', 
                        'agent_age', 'agent_rating', 'area', 'category']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Data type conversions
        df['delivery_time'] = pd.to_numeric(df['delivery_time'], errors='coerce')
        df['agent_age'] = pd.to_numeric(df['agent_age'], errors='coerce')
        df['agent_rating'] = pd.to_numeric(df['agent_rating'], errors='coerce')
        
        # Standardize categorical columns
        categorical_cols = ['traffic', 'weather', 'vehicle', 'area', 'category']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
                df[col] = df[col].replace('Nan', np.nan)
        
        # Drop rows with missing critical values
        df = df.dropna(subset=['delivery_time'])
        
        # Fill missing values
        if df['agent_age'].isnull().sum() > 0:
            df['agent_age'].fillna(df['agent_age'].median(), inplace=True)
        if df['agent_rating'].isnull().sum() > 0:
            df['agent_rating'].fillna(df['agent_rating'].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        # Derive new metrics
        mean_time = df['delivery_time'].mean()
        std_time = df['delivery_time'].std()
        threshold = mean_time + std_time
        df['is_late'] = (df['delivery_time'] > threshold).astype(int)
        
        # Age groups
        df['age_group'] = pd.cut(df['agent_age'], 
                                 bins=[0, 25, 40, 100], 
                                 labels=['<25', '25-40', '40+'])
        
        # Time categories
        df['time_category'] = pd.cut(df['delivery_time'],
                                      bins=[0, 20, 30, 40, np.inf],
                                      labels=['Very Fast (<20)', 'Fast (20-30)', 
                                             'Average (30-40)', 'Slow (>40)'])
        
        # Convert any columns that contain pandas Timestamp objects to strings
        # to avoid PyArrow serialization errors when Streamlit attempts to render dataframes.
        for col in df.columns:
            try:
                if df[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    df[col] = df[col].astype(str)
            except Exception:
                # If apply fails for large columns or mixed types, skip conversion for that column
                continue

        cleaned_shape = df.shape
        
        return df, original_shape, cleaned_shape, threshold
        
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Ensure 'Last mile Delivery Data.xlsx' is in 'data/' folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# ==================== VISUALIZATION FUNCTIONS ====================

def create_delay_analyzer(df_filtered):
    """
    Compulsory Visual 1: Delay Analyzer - Weather & Traffic impact
    """
    weather_agg = df_filtered.groupby('weather').agg({
        'delivery_time': 'mean',
        'is_late': 'mean'
    }).reset_index()
    weather_agg['late_pct'] = weather_agg['is_late'] * 100
    weather_agg = weather_agg.sort_values('delivery_time', ascending=False)
    
    traffic_agg = df_filtered.groupby('traffic').agg({
        'delivery_time': 'mean',
        'is_late': 'mean'
    }).reset_index()
    traffic_agg['late_pct'] = traffic_agg['is_late'] * 100
    traffic_agg = traffic_agg.sort_values('delivery_time', ascending=False)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Delivery Time by Weather', 
                       'Average Delivery Time by Traffic'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    fig.add_trace(
        go.Bar(x=weather_agg['weather'], 
               y=weather_agg['delivery_time'],
               name='Avg Time',
               marker_color='lightblue',
               text=weather_agg['delivery_time'].round(1),
               textposition='outside',
               showlegend=True),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=weather_agg['weather'],
                  y=weather_agg['late_pct'],
                  name='Late %',
                  mode='lines+markers',
                  marker=dict(size=10, color='red'),
                  line=dict(width=2, color='red'),
                  showlegend=True),
        row=1, col=1, secondary_y=True
    )
    
    fig.add_trace(
        go.Bar(x=traffic_agg['traffic'],
               y=traffic_agg['delivery_time'],
               name='Avg Time',
               marker_color='lightgreen',
               text=traffic_agg['delivery_time'].round(1),
               textposition='outside',
               showlegend=False),
        row=1, col=2, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=traffic_agg['traffic'],
                  y=traffic_agg['late_pct'],
                  name='Late %',
                  mode='lines+markers',
                  marker=dict(size=10, color='red'),
                  line=dict(width=2, color='red'),
                  showlegend=False),
        row=1, col=2, secondary_y=True
    )
    
    fig.update_yaxes(title_text="Avg Delivery Time (min)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="% Late Deliveries", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Avg Delivery Time (min)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="% Late Deliveries", row=1, col=2, secondary_y=True)
    
    fig.update_layout(height=400, showlegend=True, 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

def create_vehicle_comparison(df_filtered):
    """
    Compulsory Visual 2: Vehicle Performance Comparison
    """
    vehicle_agg = df_filtered.groupby('vehicle').agg({
        'delivery_time': 'mean',
        'is_late': 'mean'
    }).reset_index()
    vehicle_agg['late_pct'] = vehicle_agg['is_late'] * 100
    vehicle_agg = vehicle_agg.sort_values('delivery_time')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=vehicle_agg['vehicle'],
        y=vehicle_agg['delivery_time'],
        text=vehicle_agg['delivery_time'].round(1),
        textposition='outside',
        marker=dict(
            color=vehicle_agg['delivery_time'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Avg Time")
        ),
        hovertemplate='<b>%{x}</b><br>Avg Time: %{y:.1f} min<br>Late: ' + 
                     vehicle_agg['late_pct'].round(1).astype(str) + '%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Vehicle Performance Comparison",
        xaxis_title="Vehicle Type",
        yaxis_title="Average Delivery Time (minutes)",
        height=400
    )
    
    return fig

def create_agent_performance_scatter(df_filtered):
    """
    Compulsory Visual 3: Agent Performance Scatter with Manual Trendline
    ✅ NO statsmodels - Uses numpy polyfit instead
    """
    # Create scatter WITHOUT trendline parameter
    fig = px.scatter(
        df_filtered,
        x='agent_rating',
        y='delivery_time',
        color='age_group',
        size='delivery_time',
        hover_data=['vehicle', 'area', 'category'],
        title='Agent Performance: Rating vs Delivery Time by Age Group',
        labels={
            'agent_rating': 'Agent Rating',
            'delivery_time': 'Delivery Time (min)',
            'age_group': 'Age Bin'
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Manually add trendline using numpy (NO dependencies needed)
    valid_data = df_filtered[['agent_rating', 'delivery_time']].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data['agent_rating'], valid_data['delivery_time'], 1)
        p = np.poly1d(z)
        
        x_trend = np.linspace(valid_data['agent_rating'].min(), 
                             valid_data['agent_rating'].max(), 100)
        y_trend = p(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name='Trendline',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=True,
            hovertemplate='Trendline<extra></extra>'
        ))
    
    fig.update_layout(height=450)
    return fig

def create_area_heatmap(df_filtered):
    """
    Compulsory Visual 4: Area × Category Heatmap
    """
    pivot = df_filtered.pivot_table(
        values='delivery_time',
        index='area',
        columns='category',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot.values, 1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Avg Time (min)")
    ))
    
    fig.update_layout(
        title='Delivery Time Heatmap: Area × Category',
        xaxis_title='Product Category',
        yaxis_title='Delivery Area',
        height=500
    )
    
    return fig

def create_category_boxplot(df_filtered):
    """
    Compulsory Visual 5: Category Distribution Boxplot
    """
    fig = px.box(
        df_filtered,
        x='category',
        y='delivery_time',
        color='category',
        title='Delivery Time Distribution by Product Category',
        labels={
            'category': 'Product Category',
            'delivery_time': 'Delivery Time (min)'
        },
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(height=450, showlegend=False)
    return fig

def create_time_distribution(df_filtered):
    """Optional: Delivery time distribution"""
    fig = px.histogram(
        df_filtered,
        x='delivery_time',
        nbins=30,
        title='Distribution of Delivery Times',
        labels={'delivery_time': 'Delivery Time (min)', 'count': 'Frequency'},
        color_discrete_sequence=['steelblue']
    )
    fig.update_layout(height=350)
    return fig

def create_late_delivery_analysis(df_filtered):
    """Optional: Late delivery analysis"""
    late_by_traffic = df_filtered.groupby('traffic')['is_late'].mean() * 100
    late_by_weather = df_filtered.groupby('weather')['is_late'].mean() * 100
    
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Late Deliveries by Traffic', 
                                     'Late Deliveries by Weather'))
    
    fig.add_trace(go.Bar(x=late_by_traffic.index, y=late_by_traffic.values,
                        marker_color='coral', showlegend=False),
                 row=1, col=1)
    fig.add_trace(go.Bar(x=late_by_weather.index, y=late_by_weather.values,
                        marker_color='lightseagreen', showlegend=False),
                 row=1, col=2)
    
    fig.update_yaxes(title_text="% Late Deliveries", row=1, col=1)
    fig.update_yaxes(title_text="% Late Deliveries", row=1, col=2)
    fig.update_layout(height=350, title_text="Late Delivery Analysis")
    
    return fig

def create_agent_count_by_area(df_filtered):
    """Optional: Deliveries by area"""
    agent_count = df_filtered.groupby('area').size().reset_index(name='count')
    agent_count = agent_count.sort_values('count', ascending=False)
    
    fig = px.bar(agent_count, x='area', y='count',
                title='Number of Deliveries by Area',
                labels={'area': 'Area', 'count': 'Number of Deliveries'},
                color='count', color_continuous_scale='Blues')
    fig.update_layout(height=350)
    return fig

# ==================== MAIN APP ====================

def render_auth_page():
    st.title("Delivery Analytics Platform")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login(username, password):
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Sign Up")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if signup(new_username, new_password):
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists")

def render_data_filters(df: pd.DataFrame):
    st.subheader("Select Data Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_weather = st.multiselect(
            "Weather Condition",
            options=sorted(df['weather'].unique().tolist()),
            default=sorted(df['weather'].unique().tolist()),
            key="weather_filter"
        )
        
        selected_traffic = st.multiselect(
            "Traffic Level",
            options=sorted(df['traffic'].unique().tolist()),
            default=sorted(df['traffic'].unique().tolist()),
            key="traffic_filter"
        )
        
        selected_vehicle = st.multiselect(
            "Vehicle Type",
            options=sorted(df['vehicle'].unique().tolist()),
            default=sorted(df['vehicle'].unique().tolist()),
            key="vehicle_filter"
        )
    
    with col2:
        selected_area = st.multiselect(
            "Delivery Area",
            options=sorted(df['area'].unique().tolist()),
            default=sorted(df['area'].unique().tolist()),
            key="area_filter"
        )
        
        selected_category = st.multiselect(
            "Product Category",
            options=sorted(df['category'].unique().tolist()),
            default=sorted(df['category'].unique().tolist()),
            key="category_filter"
        )
    
    if st.button("Apply Filters", type="primary"):
        add_to_history("Applied filters")
        return selected_weather, selected_traffic, selected_vehicle, selected_area, selected_category
    return None, None, None, None, None

def main():
    if not st.session_state.user_logged_in:
        render_auth_page()
        return

    st.title("Advanced Delivery Analytics Platform")
    
    # Sidebar: compact account + history actions
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:8px'>
            <h3 style='margin:0'>Account</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"**User:** {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.user_logged_in = False
            st.session_state.username = None
            st.rerun()

        st.markdown("---")
        st.write("**History**")
        if st.button("Delete History"):
            if st.session_state.username in st.session_state.history:
                st.session_state.history[st.session_state.username] = []
            st.success("History cleared successfully!")
        st.caption("Note: History is stored locally in this session file.")
    
    # (History viewer removed from main area - use Delete History in sidebar)
    
    st.markdown("---")
    
    # Main content area
    st.markdown("""
    <div style='background-color: var(--metric-bg); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h3 style='margin:0'>Analytics Dashboard</h3>
        <p style='margin:0.5rem 0 0 0'>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

    # Load data
    try:
        with st.spinner("Initializing dashboard..."):
            data_path = Path(__file__).parent / "Last mile Delivery Data.xlsx"
            df, original_shape, cleaned_shape, late_threshold = load_and_clean_data(str(data_path))
            st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Data filters in main area with modern design
    with st.expander("🔍 Advanced Data Filters", expanded=False):
        st.markdown("""
        <style>
        .filter-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_weather = st.multiselect(
                "🌤️ Weather Condition",
                options=sorted(df['weather'].unique().tolist()),
                default=sorted(df['weather'].unique().tolist())
            )
            
            selected_traffic = st.multiselect(
                "🚦 Traffic Level",
                options=sorted(df['traffic'].unique().tolist()),
                default=sorted(df['traffic'].unique().tolist())
            )
            
        with col2:
            selected_vehicle = st.multiselect(
                "🚗 Vehicle Type",
                options=sorted(df['vehicle'].unique().tolist()),
                default=sorted(df['vehicle'].unique().tolist())
            )
            
            selected_area = st.multiselect(
                "📍 Delivery Area",
                options=sorted(df['area'].unique().tolist()),
                default=sorted(df['area'].unique().tolist())
            )
            
        with col3:
            selected_category = st.multiselect(
                "📦 Product Category",
                options=sorted(df['category'].unique().tolist()),
                default=sorted(df['category'].unique().tolist())
            )
            
        if st.button("Apply Filters", type="primary"):
            add_to_history("Applied data filters")

    # Apply filters
    df_filtered = df[
        df['weather'].isin(selected_weather) &
        df['traffic'].isin(selected_traffic) &
        df['vehicle'].isin(selected_vehicle) &
        df['area'].isin(selected_area) &
        df['category'].isin(selected_category)
    ]
    
    st.markdown("---")

    # KPIs — concise and clean
    st.markdown("<div class='carded'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_time = df_filtered['delivery_time'].mean()
        st.metric("Average Delivery Time", f"{avg_time:.1f} min", delta=None)
    with col2:
        late_pct = (df_filtered['is_late'].mean() * 100)
        st.metric("Late Deliveries", f"{late_pct:.1f}%", delta=None)
    with col3:
        total_deliveries = len(df_filtered)
        st.metric("Total Deliveries", f"{total_deliveries:,}", delta=None)
    with col4:
        avg_rating = df_filtered['agent_rating'].mean()
        st.metric("Avg Agent Rating", f"{avg_rating:.2f}", delta=None)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Create main tabs for analysis categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Weather & Traffic Impact",
        "🚗 Vehicle Performance",
        "👥 Agent Insights",
        "🗺️ Geographic Analysis",
        "📦 Category Analysis"
    ])

    with tab1:
        st.subheader("Weather & Traffic Impact Analysis")
        fig = create_delay_analyzer(df_filtered)
        st.plotly_chart(fig, use_container_width=True, key="delay_analyzer")

        # Single CSV download related to this visualization
        weather_df = df_filtered.groupby('weather').agg(avg_time=('delivery_time','mean'), late_pct=('is_late', 'mean')).reset_index()
        traffic_df = df_filtered.groupby('traffic').agg(avg_time=('delivery_time','mean'), late_pct=('is_late', 'mean')).reset_index()
        weather_df['dimension'] = 'weather'
        traffic_df['dimension'] = 'traffic'
        csv_data = pd.concat([weather_df, traffic_df], ignore_index=True)

        st.download_button(
            label="📥 Download CSV for Weather & Traffic",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name="weather_traffic.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("Vehicle Fleet Performance Analysis")
        fig = create_vehicle_comparison(df_filtered)
        st.plotly_chart(fig, use_container_width=True, key="vehicle_comparison")

        # CSV for vehicle performance
        csv_data = df_filtered.groupby('vehicle').agg(
            avg_time=('delivery_time','mean'),
            late_pct=('is_late','mean'),
            avg_rating=('agent_rating','mean'),
            deliveries=('delivery_time','count')
        ).reset_index().round(2)

        st.download_button(
            label="📥 Download CSV for Vehicle Performance",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name="vehicle_performance.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Agent Performance Analytics")
        fig = create_agent_performance_scatter(df_filtered)
        st.plotly_chart(fig, use_container_width=True, key="agent_performance")

        # CSV for agent analysis (summary)
        csv_data = df_filtered.groupby('age_group').agg(
            avg_time=('delivery_time','mean'),
            avg_rating=('agent_rating','mean'),
            deliveries=('delivery_time','count'),
            late_pct=('is_late','mean')
        ).reset_index().round(2)

        st.download_button(
            label="📥 Download CSV for Agent Performance",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name="agent_performance.csv",
            mime="text/csv"
        )

    with tab4:
        st.subheader("Geographic Delivery Analysis")
        fig = create_area_heatmap(df_filtered)
        st.plotly_chart(fig, use_container_width=True, key="area_heatmap")

        csv_data = df_filtered.groupby('area').agg(
            avg_time=('delivery_time','mean'),
            late_pct=('is_late','mean'),
            deliveries=('delivery_time','count'),
            avg_rating=('agent_rating','mean')
        ).reset_index().round(2)

        st.download_button(
            label="📥 Download CSV for Geographic Analysis",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name="geographic_analysis.csv",
            mime="text/csv"
        )

    with tab5:
        st.subheader("Product Category Performance")
        fig = create_category_boxplot(df_filtered)
        st.plotly_chart(fig, use_container_width=True, key="category_boxplot")

        csv_data = df_filtered.groupby('category').agg(
            avg_time=('delivery_time','mean'),
            late_pct=('is_late','mean'),
            deliveries=('delivery_time','count')
        ).reset_index().round(2)

        st.download_button(
            label="📥 Download CSV for Category Analysis",
            data=csv_data.to_csv(index=False).encode('utf-8'),
            file_name="category_analysis.csv",
            mime="text/csv"
        )
    
    # (Removed duplicate data reload and sidebar filter block to avoid duplication.)
    
    # ==================== OPTIONAL VISUALIZATIONS ====================
    
    with st.expander("🔍 Additional Insights (Optional Visuals)", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_time_distribution(df_filtered), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_late_delivery_analysis(df_filtered), use_container_width=True)
        
        st.plotly_chart(create_agent_count_by_area(df_filtered), use_container_width=True)
    
    # ==================== EXPORT FILTERED SUMMARIES ====================
    
    st.markdown("---")
    st.subheader("📤 Export Filtered Summaries")
    
    export_data = {
        'by_traffic': df_filtered.groupby('traffic').agg({
            'delivery_time': 'mean',
            'is_late': lambda x: (x.mean() * 100)
        }).reset_index().rename(columns={'delivery_time': 'avg_time', 'is_late': 'late_pct'}),
        
        'by_weather': df_filtered.groupby('weather').agg({
            'delivery_time': 'mean',
            'is_late': lambda x: (x.mean() * 100)
        }).reset_index().rename(columns={'delivery_time': 'avg_time', 'is_late': 'late_pct'}),
        
        'by_vehicle': df_filtered.groupby('vehicle').agg({
            'delivery_time': 'mean',
            'is_late': lambda x: (x.mean() * 100)
        }).reset_index().rename(columns={'delivery_time': 'avg_time', 'is_late': 'late_pct'}),
        
        'by_area': df_filtered.groupby('area').agg({
            'delivery_time': 'mean',
            'is_late': lambda x: (x.mean() * 100)
        }).reset_index().rename(columns={'delivery_time': 'avg_time', 'is_late': 'late_pct'}),
        
        'by_category': df_filtered.groupby('category').agg({
            'delivery_time': 'mean',
            'is_late': lambda x: (x.mean() * 100)
        }).reset_index().rename(columns={'delivery_time': 'avg_time', 'is_late': 'late_pct'})
    }
    
    all_df = pd.concat(export_data, names=['group']).reset_index()
    
    csv_bytes = all_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV snapshot",
        data=csv_bytes,
        file_name="filtered_delivery_summaries.csv",
        mime="text/csv"
    )
    
    st.caption("💡 Download aggregated metrics for all filtered dimensions.")
    
    # ==================== DATA QUALITY INFO ====================
    
    with st.expander("ℹ️ Data Quality Report"):
        st.write("**Original Dataset:**", f"{original_shape[0]} rows × {original_shape[1]} columns")
        st.write("**After Cleaning:**", f"{cleaned_shape[0]} rows × {cleaned_shape[1]} columns")
        st.write("**Rows Removed:**", original_shape[0] - cleaned_shape[0])
        
        st.write("\n**Summary Statistics:**")
        st.dataframe(df_filtered.describe())
        
        st.write("\n**Categorical Value Counts:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Weather:**", dict(df_filtered['weather'].value_counts()))
            st.write("**Traffic:**", dict(df_filtered['traffic'].value_counts()))
            st.write("**Vehicle:**", dict(df_filtered['vehicle'].value_counts()))
        with col2:
            st.write("**Area:**", dict(df_filtered['area'].value_counts()))
            st.write("**Category:**", dict(df_filtered['category'].value_counts()))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Made by Pravar Golecha</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected error in main application")