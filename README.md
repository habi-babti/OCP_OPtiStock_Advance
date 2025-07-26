# OCP OptiStock - Inventory Forecasting System

## Project Overview

OCP OptiStock is a comprehensive inventory management and demand forecasting system built with Streamlit. Designed specifically for industrial operations like OCP (Office Chérifien des Phosphates), this application provides real-time inventory tracking, AI-powered demand forecasting, and intelligent stock optimization recommendations.

The system combines traditional statistical methods with AI-powered insights to help businesses maintain optimal stock levels, reduce costs, and prevent stockouts.

## Key Features

### 📊 **Dashboard & Analytics**
- Real-time inventory overview with key metrics
- Interactive charts showing stock distribution by category
- Low stock alerts and threshold monitoring
- Historical stock movement analysis

### 🔐 **User Management**
- Role-based authentication (Admin, Manager, User)
- Secure password hashing
- Permission-based access control for sensitive operations

### 📦 **Inventory Management**
- Add, edit, and delete inventory items
- Track stock levels, thresholds, and unit costs
- Supplier information management
- Category-based organization

### 🔮 **AI-Powered Forecasting**
- Demand prediction using linear regression and AI models
- Integration with Ollama for advanced AI insights
- Confidence intervals and trend analysis
- Risk assessment and reorder recommendations

### 📈 **Advanced Analytics**
- Stock movement trend analysis
- Cost analysis by category and item
- Movement type distribution (in/out/adjustment)
- High-value item identification

### 📋 **Data Management**
- CSV import/export functionality
- Historical data tracking
- Automated sample data generation
- Data validation and error handling

### 🤖 **AI Insights**
- Stock optimization recommendations
- Anomaly detection in stock movements
- Automated reorder suggestions
- Cost reduction opportunities identification

## How Forecasting Works

The forecasting system operates through a multi-layered approach:

### 1. **Data Collection**
- Retrieves historical stock data for the selected item
- Analyzes recent stock movements and trends
- Considers current stock levels and thresholds

### 2. **Forecasting Methods**

#### **Simple Forecasting (Fallback)**
- Uses linear regression on recent stock levels (14-day window)
- Extrapolates trends for the specified forecast period
- Provides basic risk assessment based on threshold proximity

#### **AI-Powered Forecasting**
- Integrates with Ollama AI models (llama3.1, gemma3, deepseek-coder-v2)
- Analyzes complex patterns in historical data
- Considers item category, seasonality, and business context
- Generates JSON responses with predictions, trends, and recommendations

### 3. **Enhanced Features**
- **Confidence Bands**: Statistical uncertainty visualization
- **Exponential Smoothing**: Reduces forecast noise
- **Threshold Analysis**: Identifies critical stock periods
- **Risk Categorization**: Low/Medium/High risk levels based on predictions

### 4. **Output Generation**
- Interactive Plotly charts with threshold lines
- Categorized recommendations (Actions, Alerts, Insights)
- Statistical summaries and analytics
- Time-sensitive alerts for critical stock levels

## Dependencies

### Core Libraries
```python
streamlit>=1.28.0          # Web application framework
pandas>=1.5.0              # Data manipulation and analysis
numpy>=1.24.0              # Numerical computing
plotly>=5.15.0             # Interactive visualizations
scikit-learn>=1.3.0        # Machine learning algorithms
requests>=2.31.0           # HTTP requests for AI integration
```

### Additional Requirements
- **Ollama** (optional): Local AI model server for advanced forecasting
  - Install from: https://ollama.ai/
  - Recommended models: `llama3.1:latest`, `gemma3:12b`
- **Python 3.8+**: Required for all features

## Usage Instructions

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd ocp-optistock

# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn requests

# Optional: Install and setup Ollama for AI features
# Follow instructions at https://ollama.ai/
```

### 2. **Running the Application**
```bash
# Start the Streamlit application
streamlit run app_streamlit_proto_type.py

# Access the application at http://localhost:8501
```

### 3. **Default Login Credentials**
```
Admin:   admin / admin123
Manager: manager / manager123  
User:    user / user123
```

### 4. **Data Structure**
The application automatically creates the following data files:
- `data/inventory.csv` - Main inventory database
- `data/users.json` - User authentication data
- `data/stock_history.csv` - Historical stock movements

### 5. **AI Integration Setup (Optional)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull llama3.1:latest
ollama pull gemma3:12b

# Start Ollama server (runs on localhost:11434)
ollama serve
```

## File Structure

```
project/
├── app_streamlit_proto_type.py    # Main application file
├── data/                          # Auto-generated data directory
│   ├── inventory.csv             # Inventory database
│   ├── users.json               # User credentials
│   └── stock_history.csv        # Historical data
├── templates/                    # Static assets
│   └── ocp_png.png              # Logo file
└── README.md                    # This file
```

## Integration Notes

- **Database**: Currently uses CSV files; can be extended to PostgreSQL/MySQL
- **Authentication**: Basic hash-based system; can integrate with LDAP/OAuth
- **AI Models**: Designed for Ollama but can adapt to OpenAI API or other providers
- **Deployment**: Ready for Docker containerization and cloud deployment

## Contributing

This system is designed to be modular and extensible. Key areas for enhancement:
- Database backend integration
- Advanced forecasting algorithms
- Mobile-responsive design improvements
- Multi-language support
- Advanced reporting features

---

**Note**: This is a prototype system. For production use, consider implementing additional security measures, database optimization, and comprehensive testing.
