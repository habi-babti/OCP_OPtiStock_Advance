import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import hashlib
import json
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="OCP OptiStock",
    page_icon='templates/ocp_png.png',
    layout="wide",
    initial_sidebar_state="expanded"
)


class OCPOptiStock:
    def __init__(self):
        self.data_dir = "data"
        self.inventory_file = os.path.join(self.data_dir, "inventory.csv")
        self.users_file = os.path.join(self.data_dir, "users.json")
        self.history_file = os.path.join(self.data_dir, "stock_history.csv")
        self.ensure_data_structure()
        self.ollama_base_url = "http://localhost:11434"

    def ensure_data_structure(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(self.inventory_file):
            sample_data = pd.DataFrame({
                'item_id': ['ITM001', 'ITM002', 'ITM003', 'ITM004', 'ITM005'],
                'item_name': ['Phosphate Rock', 'Sulfuric Acid', 'Ammonia', 'Safety Equipment', 'Mining Tools'],
                'category': ['Raw Materials', 'Chemicals', 'Chemicals', 'Safety', 'Equipment'],
                'current_stock': [15000, 2500, 1800, 450, 120],
                'unit': ['Tons', 'Liters', 'Liters', 'Units', 'Units'],
                'min_threshold': [5000, 500, 300, 100, 50],
                'max_threshold': [20000, 5000, 3000, 1000, 200],
                'unit_cost': [85.5, 1.2, 2.8, 125.0, 350.0],
                'supplier': ['Local Mine A', 'ChemCorp', 'ChemCorp', 'SafetyFirst', 'ToolsInc'],
                'last_updated': [datetime.now().strftime('%Y-%m-%d')] * 5
            })
            sample_data.to_csv(self.inventory_file, index=False)

        if not os.path.exists(self.users_file):
            users_data = {
                "admin": {"password": self.hash_password("admin123"), "role": "admin"},
                "manager": {"password": self.hash_password("manager123"), "role": "manager"},
                "user": {"password": self.hash_password("user123"), "role": "user"}
            }
            with open(self.users_file, 'w') as f:
                json.dump(users_data, f)

        if not os.path.exists(self.history_file):
            self.generate_sample_history()

    def generate_sample_history(self):
        items = ['ITM001', 'ITM002', 'ITM003', 'ITM004', 'ITM005']
        dates = pd.date_range(start=datetime.now() - timedelta(days=90),
                              end=datetime.now(), freq='D')

        history_data = []
        for item in items:
            base_stock = np.random.randint(1000, 5000)
            for date in dates:
                variation = np.random.normal(0, 100)
                stock_level = max(0, base_stock + variation + np.random.randint(-200, 200))
                history_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'item_id': item,
                    'stock_level': int(stock_level),
                    'movement_type': np.random.choice(['in', 'out', 'adjustment'], p=[0.3, 0.5, 0.2]),
                    'quantity': np.random.randint(50, 500)
                })

        pd.DataFrame(history_data).to_csv(self.history_file, index=False)

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate_user(self, username, password):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            if username in users and users[username]['password'] == self.hash_password(password):
                return users[username]['role']
        return None

    def load_inventory(self):
        df = pd.read_csv(self.inventory_file)
        required_columns = ['item_id', 'item_name', 'category', 'current_stock',
                            'unit', 'min_threshold', 'max_threshold', 'unit_cost',
                            'supplier', 'last_updated']

        # Add missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ['current_stock', 'min_threshold', 'max_threshold']:
                    df[col] = 0
                elif col == 'unit_cost':
                    df[col] = 0.0
                elif col == 'last_updated':
                    df[col] = datetime.now().strftime('%Y-%m-%d')
                else:
                    df[col] = 'Unknown'

        # Ensure numeric columns are properly typed
        numeric_columns = ['current_stock', 'min_threshold', 'max_threshold', 'unit_cost']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def save_inventory(self, df):
        df.to_csv(self.inventory_file, index=False)

    def load_history(self):
        try:
            return pd.read_csv(self.history_file)
        except FileNotFoundError:
            self.generate_sample_history()
            return pd.read_csv(self.history_file)

    def get_low_stock_items(self):
        try:
            df = self.load_inventory()
            # Ensure we have the required columns
            if 'current_stock' in df.columns and 'min_threshold' in df.columns:
                return df[df['current_stock'] <= df['min_threshold']]
            else:
                return pd.DataFrame()  # Return empty DataFrame if columns missing
        except Exception as e:
            st.error(f"Error loading inventory data: {str(e)}")
            return pd.DataFrame()

    def query_ollama(self, model, prompt):
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', 'No response received')
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Connection error: {str(e)}"

    def forecast_with_ai(self, item_data, periods=30):
        try:
            history = self.load_history()
            item_history = history[history['item_id'] == item_data['item_id']].sort_values('date')

            if len(item_history) < 10:
                return self.simple_forecast(item_data, periods)

            prompt = f"""
            Analyze the following inventory data for {item_data['item_name']} and provide a forecast:

            Recent stock levels: {item_history['stock_level'].tail(14).tolist()}
            Current stock: {item_data['current_stock']}
            Category: {item_data['category']}
            Min threshold: {item_data['min_threshold']}

            Provide a JSON response with:
            1. forecast: array of {periods} predicted stock levels
            2. trend: "increasing", "decreasing", or "stable"
            3. risk_level: "low", "medium", or "high"
            4. recommendation: brief text recommendation

            Return only valid JSON.
            """

            ai_response = self.query_ollama("llama3.1:latest", prompt)

            try:
                import json
                result = json.loads(ai_response)
                return result
            except:
                return self.simple_forecast(item_data, periods)

        except Exception as e:
            return self.simple_forecast(item_data, periods)

    def simple_forecast(self, item_data, periods=30):
        try:
            history = self.load_history()
            item_history = history[history['item_id'] == item_data['item_id']].sort_values('date')

            if len(item_history) < 5:
                current_stock = item_data.get('current_stock', 1000)
                trend_forecast = [current_stock] * periods
            else:
                recent_levels = item_history['stock_level'].tail(14).values
                X = np.arange(len(recent_levels)).reshape(-1, 1)
                y = recent_levels

                model = LinearRegression()
                model.fit(X, y)

                future_X = np.arange(len(recent_levels), len(recent_levels) + periods).reshape(-1, 1)
                trend_forecast = model.predict(future_X).tolist()

            current_stock = item_data.get('current_stock', 1000)
            min_threshold = item_data.get('min_threshold', 500)

            risk_level = "low"
            if current_stock <= min_threshold:
                risk_level = "high"
            elif current_stock <= min_threshold * 1.5:
                risk_level = "medium"

            return {
                'forecast': [max(0, int(f)) for f in trend_forecast],
                'trend': 'stable',
                'risk_level': risk_level,
                'recommendation': f"Monitor stock levels closely. Current stock: {current_stock}"
            }
        except Exception as e:
            # Fallback forecast if there's any error
            current_stock = item_data.get('current_stock', 1000)
            return {
                'forecast': [current_stock] * periods,
                'trend': 'stable',
                'risk_level': 'medium',
                'recommendation': f"Basic forecast due to data limitations. Current stock: {current_stock}"
            }


def main():
    app = OCPOptiStock()

    st.markdown("""
    <style>
    :root {
        --primary-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        --secondary-gradient: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        --danger-gradient: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.1), 0 5px 10px rgba(0,0,0,0.05);
        --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    .main-header {
        background: var(--primary-gradient);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }

    .main-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: rgba(255,255,255,0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
        transition: var(--transition);
        border-left: 4px solid transparent;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }

    .metric-card.primary {
        border-left-color: #2a5298;
    }

    .metric-card.success {
        border-left-color: #38ef7d;
    }

    .metric-card.warning {
        border-left-color: #ff9a44;
    }

    .metric-card.danger {
        border-left-color: #ff4b2b;
    }

    .low-stock-alert {
        background: linear-gradient(90deg, #fff8f8 0%, #ffebee 100%);
        border-left: 4px solid #f44336;
        padding: 1.25rem;
        margin: 1.25rem 0;
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: var(--transition);
    }

    .low-stock-alert:hover {
        transform: translateX(5px);
    }

    .low-stock-alert::before {
        content: '⚠️';
        font-size: 1.5rem;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: var(--shadow-md);
        border-radius: 0 12px 12px 0;
    }

    /* Responsive enhancements */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            border-radius: 0;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.75rem 0;
        }
    }

    /* Animation for critical alerts */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }

    .critical-alert {
        animation: pulse 2s infinite;
        background: var(--danger-gradient);
        color: white;
    }
</style>
    """, unsafe_allow_html=True)

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None

    if not st.session_state.authenticated:
        st.markdown('<div class="main-header"><h1>OCP OptiStock</h1><p>Inventory Forecasting System</p></div>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                role = app.authenticate_user(username, password)
                if role:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.info("Demo credentials: admin/admin123, manager/manager123, user/user123")
        return

    st.markdown(
        f'<div class="main-header"><h1> OCP OptiStock</h1><p>Welcome, {st.session_state.username} ({st.session_state.user_role})</p></div>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("###  Navigation")

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Inventory Management", "Forecasting", "Analytics", "Data Import/Export", "AI Insights"]
        )

        st.markdown("---")
        low_stock = app.get_low_stock_items()
        if len(low_stock) > 0:
            st.markdown("### ⚠️ Low Stock Alerts")
            for _, item in low_stock.iterrows():
                item_name = item.get('item_name', 'Unknown Item')
                current_stock = item.get('current_stock', 0)
                unit = item.get('unit', 'units')
                st.error(f"{item_name}: {current_stock} {unit}")

    if page == "Dashboard":
        show_dashboard(app)
    elif page == "Inventory Management":
        show_inventory_management(app)
    elif page == "Forecasting":
        show_forecasting(app)
    elif page == "Analytics":
        show_analytics(app)
    elif page == "Data Import/Export":
        show_import_export(app)
    elif page == "AI Insights":
        show_ai_insights(app)


def show_dashboard(app):
    st.header("Dashboard Overview")

    try:
        df = app.load_inventory()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Items", len(df))
        with col2:
            if 'current_stock' in df.columns and 'unit_cost' in df.columns:
                total_value = (df['current_stock'] * df['unit_cost']).sum()
                st.metric("Total Inventory Value", f"${total_value:,.2f}")
            else:
                st.metric("Total Inventory Value", "N/A")
        with col3:
            low_stock_count = len(app.get_low_stock_items())
            st.metric("Low Stock Items", low_stock_count, delta=-low_stock_count if low_stock_count > 0 else 0)
        with col4:
            if 'category' in df.columns:
                categories = df['category'].nunique()
                st.metric("Categories", categories)
            else:
                st.metric("Categories", "N/A")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stock by Category")
            if 'category' in df.columns and 'current_stock' in df.columns:
                category_stock = df.groupby('category')['current_stock'].sum().reset_index()
                fig = px.pie(category_stock, values='current_stock', names='category',
                             title="Inventory Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Category or stock data not available")

        with col2:
            st.subheader("Stock Levels vs Thresholds")
            required_cols = ['item_name', 'current_stock', 'min_threshold']
            if all(col in df.columns for col in required_cols):
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Current Stock', x=df['item_name'], y=df['current_stock']))
                fig.add_trace(go.Scatter(name='Min Threshold', x=df['item_name'], y=df['min_threshold'],
                                         mode='markers', marker=dict(color='red', size=8)))
                fig.update_layout(title="Stock Levels vs Minimum Thresholds")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Threshold data not available")

        st.subheader("Recent Stock History")
        try:
            history = app.load_history()
            recent_history = history.tail(50)

            if 'date' in recent_history.columns and 'stock_level' in recent_history.columns:
                fig = px.line(recent_history, x='date', y='stock_level', color='item_id',
                              title="Recent Stock Level Changes")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Historical data not available")
        except Exception as e:
            st.info("Historical data not available")

    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        st.info("Please check your data files and try refreshing the page.")


def show_inventory_management(app):
    st.header("Inventory Management")

    if st.session_state.user_role in ['admin', 'manager']:
        tab1, tab2, tab3 = st.tabs(["View Inventory", "Add Item", "Edit Item"])

        with tab1:
            df = app.load_inventory()
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.subheader("Add New Item")
            with st.form("add_item"):
                col1, col2 = st.columns(2)
                with col1:
                    item_id = st.text_input("Item ID")
                    item_name = st.text_input("Item Name")
                    category = st.selectbox("Category",
                                            ["Raw Materials", "Chemicals", "Equipment", "Safety", "Other"])
                    current_stock = st.number_input("Current Stock", min_value=0)
                    unit = st.text_input("Unit")

                with col2:
                    min_threshold = st.number_input("Minimum Threshold", min_value=0)
                    max_threshold = st.number_input("Maximum Threshold", min_value=0)
                    unit_cost = st.number_input("Unit Cost", min_value=0.0, format="%.2f")
                    supplier = st.text_input("Supplier")

                if st.form_submit_button("Add Item"):
                    if item_id and item_name:
                        df = app.load_inventory()
                        new_item = pd.DataFrame({
                            'item_id': [item_id],
                            'item_name': [item_name],
                            'category': [category],
                            'current_stock': [current_stock],
                            'unit': [unit],
                            'min_threshold': [min_threshold],
                            'max_threshold': [max_threshold],
                            'unit_cost': [unit_cost],
                            'supplier': [supplier],
                            'last_updated': [datetime.now().strftime('%Y-%m-%d')]
                        })
                        df = pd.concat([df, new_item], ignore_index=True)
                        app.save_inventory(df)
                        st.success("Item added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")

        with tab3:
            df = app.load_inventory()
            selected_item = st.selectbox("Select Item to Edit", df['item_name'].tolist())

            if selected_item:
                item_data = df[df['item_name'] == selected_item].iloc[0]

                with st.form("edit_item"):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_stock = st.number_input("Current Stock", value=int(item_data['current_stock']))
                        new_min = st.number_input("Min Threshold", value=int(item_data['min_threshold']))
                        new_max = st.number_input("Max Threshold", value=int(item_data['max_threshold']))

                    with col2:
                        new_cost = st.number_input("Unit Cost", value=float(item_data['unit_cost']), format="%.2f")
                        new_supplier = st.text_input("Supplier", value=item_data['supplier'])

                    if st.form_submit_button("Update Item"):
                        idx = df[df['item_name'] == selected_item].index[0]
                        df.loc[idx, 'current_stock'] = new_stock
                        df.loc[idx, 'min_threshold'] = new_min
                        df.loc[idx, 'max_threshold'] = new_max
                        df.loc[idx, 'unit_cost'] = new_cost
                        df.loc[idx, 'supplier'] = new_supplier
                        df.loc[idx, 'last_updated'] = datetime.now().strftime('%Y-%m-%d')
                        app.save_inventory(df)
                        st.success("Item updated successfully!")
                        st.rerun()
    else:
        st.subheader("Current Inventory")
        df = app.load_inventory()
        st.dataframe(df, use_container_width=True)
        st.info("You have read-only access to inventory data.")


def show_forecasting(app):
    """Enhanced demand forecasting with improved logic and performance."""
    st.header("Demand Forecasting")

    # Load inventory data with error handling
    try:
        df = app.load_inventory()
        if df.empty:
            st.error("No inventory data available. Please add items first.")
            return
    except Exception as e:
        st.error(f"Error loading inventory data: {str(e)}")
        return

    # Item selection with better UX
    item_options = df['item_name'].tolist()
    if not item_options:
        st.warning("No items available for forecasting.")
        return

    selected_item = st.selectbox(
        "Select Item for Forecasting",
        item_options,
        help="Choose an item to generate demand forecast"
    )

    if not selected_item:
        return

    # Get item data safely
    try:
        item_data = df[df['item_name'] == selected_item].iloc[0].to_dict()
    except (IndexError, KeyError) as e:
        st.error(f"Error retrieving item data: {str(e)}")
        return

    # Enhanced UI layout with better organization
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Forecast Settings")

        # Forecast period with validation
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="Select the number of days to forecast ahead"
        )

        # AI toggle with explanation
        use_ai = st.checkbox(
            "Use AI Forecasting",
            value=True,
            help="Enable AI-powered forecasting for more accurate predictions"
        )

        # Confidence interval option
        show_confidence = st.checkbox(
            "Show Confidence Bands",
            value=True,
            help="Display prediction confidence intervals"
        )

        # Additional forecast parameters
        with st.expander("Advanced Options"):
            seasonal_adjustment = st.checkbox("Apply Seasonal Adjustment", value=False)
            smoothing_factor = st.slider("Smoothing Factor", 0.1, 1.0, 0.3, 0.1)

    with col1:
        # Progress indicator for forecast generation
        with st.spinner("Generating forecast..."):
            try:
                # Generate forecast based on selected method
                if use_ai:
                    forecast_result = app.forecast_with_ai(item_data, forecast_days)
                else:
                    forecast_result = app.simple_forecast(item_data, forecast_days)

                # Validate forecast result
                if not forecast_result or 'forecast' not in forecast_result:
                    st.error("Failed to generate forecast. Using fallback method.")
                    forecast_result = app.simple_forecast(item_data, forecast_days)

                # Ensure forecast has correct length
                forecast_values = forecast_result.get('forecast', [])
                if len(forecast_values) != forecast_days:
                    # Pad or truncate forecast to match requested days
                    if len(forecast_values) < forecast_days:
                        last_value = forecast_values[-1] if forecast_values else item_data.get('current_stock', 0)
                        forecast_values.extend([last_value] * (forecast_days - len(forecast_values)))
                    else:
                        forecast_values = forecast_values[:forecast_days]

                # Apply smoothing if requested
                if smoothing_factor < 1.0:
                    forecast_values = apply_exponential_smoothing(forecast_values, smoothing_factor)

                # Create enhanced forecast visualization
                dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
                forecast_df = pd.DataFrame({
                    'date': dates,
                    'predicted_stock': forecast_values
                })

                # Calculate confidence bands if requested
                if show_confidence:
                    std_dev = np.std(forecast_values) if len(forecast_values) > 1 else 0
                    forecast_df['upper_bound'] = forecast_df['predicted_stock'] + (1.96 * std_dev)
                    forecast_df['lower_bound'] = np.maximum(0, forecast_df['predicted_stock'] - (1.96 * std_dev))

                # Create enhanced visualization
                fig = create_enhanced_forecast_chart(
                    forecast_df,
                    item_data,
                    selected_item,
                    show_confidence
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                return

    # Enhanced metrics display with better formatting
    st.subheader("📊 Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    # Current stock with status indicator
    current_stock = item_data.get('current_stock', 0)
    min_threshold = item_data.get('min_threshold', 0)
    unit = item_data.get('unit', 'units')

    with col1:
        # Color-coded current stock
        stock_status = "🔴" if current_stock <= min_threshold else "🟡" if current_stock <= min_threshold * 1.5 else "🟢"
        st.metric(
            "Current Stock",
            f"{current_stock:,.0f} {unit}",
            help=f"Status: {stock_status}"
        )

    with col2:
        # Enhanced trend display
        trend = forecast_result.get('trend', 'stable').title()
        trend_icons = {"Increasing": "📈", "Decreasing": "📉", "Stable": "➡️"}
        trend_colors = {"Increasing": "normal", "Decreasing": "inverse", "Stable": "off"}

        st.metric(
            "Trend",
            f"{trend_icons.get(trend, '➡️')} {trend}",
            delta_color=trend_colors.get(trend, "off")
        )

    with col3:
        # Risk level with color coding
        risk_level = forecast_result.get('risk_level', 'medium').title()
        risk_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        risk_colors = {"Low": "normal", "Medium": "normal", "High": "inverse"}

        st.metric(
            "Risk Level",
            f"{risk_icons.get(risk_level, '🟡')} {risk_level}",
            delta_color=risk_colors.get(risk_level, "off")
        )

    with col4:
        # Forecast accuracy indicator (if available)
        accuracy = forecast_result.get('accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            accuracy_display = f"{accuracy:.1f}%"
            accuracy_color = "normal" if accuracy >= 80 else "inverse" if accuracy < 60 else "off"
        else:
            accuracy_display = "N/A"
            accuracy_color = "off"

        st.metric(
            "Forecast Accuracy",
            accuracy_display,
            delta_color=accuracy_color
        )

    # Enhanced recommendations section
    st.subheader("💡 AI Recommendations")

    recommendation = forecast_result.get('recommendation', 'No recommendations available.')

    # Parse and enhance recommendations
    enhanced_recommendations = generate_enhanced_recommendations(
        item_data,
        forecast_result,
        forecast_values
    )

    # Display recommendations in organized tabs
    rec_tab1, rec_tab2, rec_tab3 = st.tabs(["📋 Actions", "⚠️ Alerts", "📈 Insights"])

    with rec_tab1:
        for action in enhanced_recommendations.get('actions', []):
            st.info(f"🔧 {action}")

    with rec_tab2:
        for alert in enhanced_recommendations.get('alerts', []):
            st.warning(f"⚠️ {alert}")

    with rec_tab3:
        for insight in enhanced_recommendations.get('insights', []):
            st.success(f"💡 {insight}")

    # Additional forecast analytics
    if st.expander("📊 Detailed Analytics", expanded=False):
        display_forecast_analytics(forecast_df, item_data, forecast_result)


def apply_exponential_smoothing(values, alpha):
    """Apply exponential smoothing to forecast values."""
    if not values or len(values) < 2:
        return values

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
        smoothed.append(smoothed_value)

    return smoothed


def create_enhanced_forecast_chart(forecast_df, item_data, selected_item, show_confidence):
    """Create an enhanced forecast visualization chart."""
    fig = go.Figure()

    # Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_stock'],
        mode='lines+markers',
        name='Forecasted Stock',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Stock:</b> %{y:,.0f}<extra></extra>'
    ))

    # Confidence bands
    if show_confidence and 'upper_bound' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(31, 119, 180, 0.2)',
            fill='tonexty',
            name='Confidence Band',
            hoverinfo='skip'
        ))

    # Threshold lines with better styling
    min_threshold = item_data.get('min_threshold', 0)
    max_threshold = item_data.get('max_threshold', 0)

    if min_threshold > 0:
        fig.add_hline(
            y=min_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="⚠️ Min Threshold",
            annotation_position="top right"
        )

    if max_threshold > 0:
        fig.add_hline(
            y=max_threshold,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text="✅ Max Threshold",
            annotation_position="bottom right"
        )

    # Current stock indicator
    current_stock = item_data.get('current_stock', 0)
    fig.add_hline(
        y=current_stock,
        line_dash="dot",
        line_color="orange",
        line_width=2,
        annotation_text="📊 Current Stock",
        annotation_position="top left"
    )

    # Enhanced layout
    fig.update_layout(
        title={
            'text': f"📈 Stock Forecast for {selected_item}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis_title=f"Stock Level ({item_data.get('unit', 'units')})",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100)
    )

    return fig


def generate_enhanced_recommendations(item_data, forecast_result, forecast_values):
    """Generate enhanced, categorized recommendations."""
    recommendations = {
        'actions': [],
        'alerts': [],
        'insights': []
    }

    current_stock = item_data.get('current_stock', 0)
    min_threshold = item_data.get('min_threshold', 0)
    max_threshold = item_data.get('max_threshold', 0)

    # Action recommendations
    if current_stock <= min_threshold:
        recommendations['actions'].append(
            f"Immediate reorder required - current stock ({current_stock}) is at or below minimum threshold ({min_threshold})"
        )

    if forecast_values:
        min_forecast = min(forecast_values)
        if min_forecast <= min_threshold:
            days_to_critical = next((i for i, val in enumerate(forecast_values) if val <= min_threshold), None)
            if days_to_critical is not None:
                recommendations['alerts'].append(
                    f"Stock will reach critical level in approximately {days_to_critical + 1} days"
                )

    # Risk-based recommendations
    risk_level = forecast_result.get('risk_level', 'medium')
    if risk_level == 'high':
        recommendations['actions'].append("Consider increasing safety stock levels")
        recommendations['actions'].append("Review supplier lead times and reliability")

    # Trend-based insights
    trend = forecast_result.get('trend', 'stable')
    if trend == 'decreasing':
        recommendations['insights'].append("Declining trend detected - monitor for demand changes")
    elif trend == 'increasing':
        recommendations['insights'].append("Growing demand trend - consider bulk purchasing opportunities")

    # Cost optimization insights
    unit_cost = item_data.get('unit_cost', 0)
    if unit_cost > 0:
        total_value = current_stock * unit_cost
        if total_value > 10000:  # High-value items
            recommendations['insights'].append(
                f"High-value inventory item (${total_value:,.2f}) - optimize stock levels carefully"
            )

    return recommendations


def display_forecast_analytics(forecast_df, item_data, forecast_result):
    """Display detailed forecast analytics."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistical Summary")
        forecast_values = forecast_df['predicted_stock'].values

        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                f"{np.mean(forecast_values):.1f}",
                f"{np.median(forecast_values):.1f}",
                f"{np.std(forecast_values):.1f}",
                f"{np.min(forecast_values):.1f}",
                f"{np.max(forecast_values):.1f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.subheader("Threshold Analysis")
        min_threshold = item_data.get('min_threshold', 0)
        max_threshold = item_data.get('max_threshold', 0)

        below_min = sum(1 for val in forecast_values if val <= min_threshold)
        above_max = sum(1 for val in forecast_values if val >= max_threshold)

        threshold_df = pd.DataFrame({
            'Condition': ['Days Below Min', 'Days Above Max', 'Days in Range'],
            'Count': [below_min, above_max, len(forecast_values) - below_min - above_max],
            'Percentage': [
                f"{(below_min / len(forecast_values) * 100):.1f}%",
                f"{(above_max / len(forecast_values) * 100):.1f}%",
                f"{((len(forecast_values) - below_min - above_max) / len(forecast_values) * 100):.1f}%"
            ]
        })
        st.dataframe(threshold_df, use_container_width=True)

def show_analytics(app):
    st.header("Analytics")

    df = app.load_inventory()
    history = app.load_history()

    tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Movement Trends", "Cost Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stock Distribution by Category")
            category_stats = df.groupby('category').agg({
                'current_stock': 'sum',
                'item_name': 'count'
            }).rename(columns={'item_name': 'item_count'})

            fig = px.bar(category_stats.reset_index(), x='category', y='current_stock',
                         title="Total Stock by Category")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Stock Status Overview")
            df['status'] = df.apply(lambda x: 'Low' if x['current_stock'] <= x['min_threshold']
            else 'High' if x['current_stock'] >= x['max_threshold']
            else 'Normal', axis=1)
            status_counts = df['status'].value_counts()

            fig = px.pie(values=status_counts.values, names=status_counts.index,
                         title="Stock Status Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Stock Movement Analysis")

        movement_summary = history.groupby(['date', 'movement_type']).agg({
            'quantity': 'sum'
        }).reset_index()

        fig = px.line(movement_summary, x='date', y='quantity', color='movement_type',
                      title="Daily Stock Movements")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Most Active Items")
            item_activity = history.groupby('item_id')['quantity'].sum().sort_values(ascending=False).head()
            st.bar_chart(item_activity)

        with col2:
            st.subheader("Movement Type Distribution")
            movement_dist = history['movement_type'].value_counts()
            st.bar_chart(movement_dist)

    with tab3:
        st.subheader("Cost Analysis")

        df['total_value'] = df['current_stock'] * df['unit_cost']

        col1, col2 = st.columns(2)

        with col1:
            category_value = df.groupby('category')['total_value'].sum().sort_values(ascending=False)
            fig = px.bar(x=category_value.index, y=category_value.values,
                         title="Inventory Value by Category")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("High-Value Items")
            high_value_items = df.nlargest(5, 'total_value')[['item_name', 'total_value']]
            st.dataframe(high_value_items, use_container_width=True)


def show_import_export(app):
    st.header("Data Import/Export")

    if st.session_state.user_role in ['admin', 'manager']:
        tab1, tab2 = st.tabs(["Import Data", "Export Data"])

        with tab1:
            st.subheader("Import Inventory Data")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")

            if uploaded_file is not None:
                try:
                    new_df = pd.read_csv(uploaded_file)
                    st.subheader("Preview of uploaded data:")
                    st.dataframe(new_df.head())

                    if st.button("Import Data"):
                        required_columns = ['item_id', 'item_name', 'category', 'current_stock',
                                            'unit', 'min_threshold', 'max_threshold', 'unit_cost', 'supplier']

                        if all(col in new_df.columns for col in required_columns):
                            new_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                            app.save_inventory(new_df)
                            st.success("Data imported successfully!")
                            st.rerun()
                        else:
                            st.error(f"Missing required columns: {set(required_columns) - set(new_df.columns)}")

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        with tab2:
            st.subheader("Export Current Inventory")
            df = app.load_inventory()

            col1, col2 = st.columns(2)

            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Inventory as CSV",
                    data=csv,
                    file_name=f"ocp_inventory_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            with col2:
                history = app.load_history()
                history_csv = history.to_csv(index=False)
                st.download_button(
                    label="Download History as CSV",
                    data=history_csv,
                    file_name=f"ocp_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        st.error("Access denied. Admin or Manager role required.")


def show_ai_insights(app):
    st.header("AI-Powered Insights")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("AI Models Available")
        models = ["llama3.1:latest", "gemma3:12b", "deepseek-coder-v2:latest"]
        selected_model = st.selectbox("Select AI Model", models)

        analysis_type = st.selectbox("Analysis Type", [
            "Stock Optimization",
            "Anomaly Detection",
            "Reorder Recommendations",
            "Cost Analysis"
        ])

    with col1:
        if st.button("Generate AI Insights", type="primary"):
            df = app.load_inventory()
            history = app.load_history()

            with st.spinner("Generating AI insights..."):
                if analysis_type == "Stock Optimization":
                    prompt = f"""
                    Analyze the following inventory data and provide optimization recommendations:

                    Current inventory summary:
                    - Total items: {len(df)}
                    - Categories: {df['category'].unique().tolist()}
                    - Low stock items: {len(app.get_low_stock_items())}
                    - Total inventory value: ${(df['current_stock'] * df['unit_cost']).sum():,.2f}

                    Provide recommendations for:
                    1. Stock level optimization
                    2. Cost reduction opportunities
                    3. Risk mitigation strategies

                    Format response in clear sections.
                    """

                elif analysis_type == "Anomaly Detection":
                    recent_movements = history.tail(50)
                    prompt = f"""
                    Analyze recent stock movements for anomalies:

                    Recent movement data:
                    {recent_movements.to_string()}

                    Identify:
                    1. Unusual stock level changes
                    2. Suspicious movement patterns
                    3. Items requiring immediate attention

                    Provide specific recommendations.
                    """

                elif analysis_type == "Reorder Recommendations":
                    low_stock = app.get_low_stock_items()
                    prompt = f"""
                    Generate reorder recommendations based on current stock levels:

                    Low stock items:
                    {low_stock.to_string() if len(low_stock) > 0 else "No low stock items"}

                    All items summary:
                    {df[['item_name', 'current_stock', 'min_threshold', 'max_threshold']].to_string()}

                    Provide:
                    1. Priority reorder list
                    2. Suggested order quantities
                    3. Timing recommendations
                    """

                else:  # Cost Analysis
                    prompt = f"""
                    Perform cost analysis on inventory data:

                    Inventory cost breakdown:
                    {df[['item_name', 'category', 'current_stock', 'unit_cost']].to_string()}

                    Total inventory value: ${(df['current_stock'] * df['unit_cost']).sum():,.2f}

                    Analyze:
                    1. Cost optimization opportunities
                    2. High-value items requiring special attention
                    3. Budget allocation recommendations
                    """

                response = app.query_ollama(selected_model, prompt)

                st.subheader("AI Analysis Results")
                st.markdown(response)

                if "error" in response.lower() or "connection" in response.lower():
                    st.warning("AI service unavailable. Showing sample analysis:")
                    st.markdown("""
                    ### Sample Analysis Results

                    **Stock Optimization Recommendations:**
                    - Monitor phosphate rock levels closely - approaching minimum threshold
                    - Consider bulk purchasing for chemicals to reduce unit costs
                    - Implement automated reorder points for critical safety equipment

                    **Key Insights:**
                    - 15% of items are currently below optimal stock levels
                    - Safety equipment shows irregular usage patterns
                    - Chemical inventory has 23% cost reduction potential through supplier optimization
                    """)


if __name__ == "__main__":
    main()
