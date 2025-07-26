import os
import hashlib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from flask import Flask, session, request, jsonify, send_file
from sklearn.linear_model import LinearRegression
import warnings
from flask import send_file




warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production


class OCPOptiStock:
    def __init__(self):
        self.data_dir = "data/data"
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
            if 'current_stock' in df.columns and 'min_threshold' in df.columns:
                return df[df['current_stock'] <= df['min_threshold']]
            else:
                return pd.DataFrame()
        except Exception as e:
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
                return json.loads(ai_response)
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
            current_stock = item_data.get('current_stock', 1000)
            return {
                'forecast': [current_stock] * periods,
                'trend': 'stable',
                'risk_level': 'medium',
                'recommendation': f"Basic forecast due to data limitations. Current stock: {current_stock}"
            }


# Initialize the app logic
ocp_app = OCPOptiStock()



# Authentication routes
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    role = ocp_app.authenticate_user(username, password)

    if role:
        session['authenticated'] = True
        session['user_role'] = role
        session['username'] = username
        return jsonify({
            'success': True,
            'role': role,
            'username': username
        })
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


# Inventory routes
@app.route('/inventory', methods=['GET'])
def get_inventory():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(ocp_app.load_inventory().to_dict(orient='records'))


@app.route('/inventory', methods=['POST'])
def add_inventory_item():
    if not session.get('authenticated') or session.get('user_role') not in ['admin', 'manager']:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.json
    df = ocp_app.load_inventory()
    new_item = {
        'item_id': data['item_id'],
        'item_name': data['item_name'],
        'category': data['category'],
        'current_stock': data['current_stock'],
        'unit': data['unit'],
        'min_threshold': data['min_threshold'],
        'max_threshold': data['max_threshold'],
        'unit_cost': data['unit_cost'],
        'supplier': data['supplier'],
        'last_updated': datetime.now().strftime('%Y-%m-%d')
    }
    df = pd.concat([df, pd.DataFrame([new_item])], ignore_index=True)
    ocp_app.save_inventory(df)
    return jsonify({'success': True})


@app.route('/inventory/<item_id>', methods=['PUT'])
def update_inventory_item(item_id):
    if not session.get('authenticated') or session.get('user_role') not in ['admin', 'manager']:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.json
    df = ocp_app.load_inventory()
    if item_id not in df['item_id'].values:
        return jsonify({'error': 'Item not found'}), 404

    idx = df.index[df['item_id'] == item_id][0]
    for key in data:
        if key in df.columns:
            df.at[idx, key] = data[key]
    df.at[idx, 'last_updated'] = datetime.now().strftime('%Y-%m-%d')
    ocp_app.save_inventory(df)
    return jsonify({'success': True})


# Dashboard routes
@app.route('/dashboard/summary', methods=['GET'])
def dashboard_summary():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    df = ocp_app.load_inventory()
    total_items = len(df)

    total_value = 0
    if 'current_stock' in df.columns and 'unit_cost' in df.columns:
        total_value = (df['current_stock'] * df['unit_cost']).sum()

    low_stock_count = len(ocp_app.get_low_stock_items())

    categories = 0
    if 'category' in df.columns:
        categories = df['category'].nunique()

    return jsonify({
        'total_items': total_items,
        'total_value': total_value,
        'low_stock_count': low_stock_count,
        'categories': categories
    })


@app.route('/dashboard/stock-by-category', methods=['GET'])
def stock_by_category():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    df = ocp_app.load_inventory()
    if 'category' in df.columns and 'current_stock' in df.columns:
        category_stock = df.groupby('category')['current_stock'].sum().reset_index()
        return jsonify(category_stock.to_dict(orient='records'))
    return jsonify([])


# Forecasting routes
@app.route('/forecast', methods=['GET'])
def get_forecast():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    item_id = request.args.get('item_id')
    use_ai = request.args.get('use_ai', 'true').lower() == 'true'
    periods = int(request.args.get('periods', 30))

    df = ocp_app.load_inventory()
    if item_id not in df['item_id'].values:
        return jsonify({'error': 'Item not found'}), 404

    item_data = df[df['item_id'] == item_id].iloc[0].to_dict()

    if use_ai:
        forecast_result = ocp_app.forecast_with_ai(item_data, periods)
    else:
        forecast_result = ocp_app.simple_forecast(item_data, periods)

    dates = pd.date_range(
        start=datetime.now(),
        periods=periods,
        freq='D'
    ).strftime('%Y-%m-%d').tolist()

    return jsonify({
        'item_id': item_id,
        'forecast': forecast_result['forecast'],
        'dates': dates,
        'trend': forecast_result['trend'],
        'risk_level': forecast_result['risk_level'],
        'recommendation': forecast_result['recommendation']
    })


# Analytics routes
@app.route('/analytics/movement', methods=['GET'])
def movement_analytics():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    history = ocp_app.load_history()
    movement_summary = history.groupby(['date', 'movement_type'])['quantity'].sum().reset_index()
    return jsonify(movement_summary.to_dict(orient='records'))


# Data routes
@app.route('/data/export/inventory', methods=['GET'])
def export_inventory():
    if not session.get('authenticated') or session.get('user_role') not in ['admin', 'manager']:
        return jsonify({'error': 'Unauthorized'}), 403

    df = ocp_app.load_inventory()
    csv = df.to_csv(index=False)
    return send_file(
        csv,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"ocp_inventory_{datetime.now().strftime('%Y%m%d')}.csv"
    )


@app.route('/data/import/inventory', methods=['POST'])
def import_inventory():
    if not session.get('authenticated') or session.get('user_role') not in ['admin', 'manager']:
        return jsonify({'error': 'Unauthorized'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        new_df = pd.read_csv(file)
        required_columns = ['item_id', 'item_name', 'category', 'current_stock',
                            'unit', 'min_threshold', 'max_threshold', 'unit_cost', 'supplier']

        if not all(col in new_df.columns for col in required_columns):
            missing = list(set(required_columns) - set(new_df.columns))
            return jsonify({'error': f'Missing columns: {", ".join(missing)}'}), 400

        new_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        ocp_app.save_inventory(new_df)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

from flask import send_file

@app.route('/')
def index():
    return send_file('templates/index.html')


# AI Insights
@app.route('/ai/insights', methods=['POST'])
def ai_insights():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    model = data.get('model', 'llama3.1:latest')
    analysis_type = data.get('analysis_type', 'Stock Optimization')

    df = ocp_app.load_inventory()
    history = ocp_app.load_history()

    if analysis_type == "Stock Optimization":
        prompt = f"""
        Analyze inventory data and provide optimization recommendations:
        Total items: {len(df)}
        Categories: {df['category'].unique().tolist()}
        Low stock items: {len(ocp_app.get_low_stock_items())}
        Total inventory value: ${(df['current_stock'] * df['unit_cost']).sum():,.2f}
        """
    # Other analysis types would be handled similarly...

    response = ocp_app.query_ollama(model, prompt)
    return jsonify({'insights': response})


if __name__ == '__main__':
    app.run(debug=True)