# SmartDynamic: AI-Powered Dynamic Pricing System

An AI-powered context-aware dynamic pricing system for e-commerce built with Python, machine learning, reinforcement learning, and explainable AI.

## Project Overview

SmartDynamic is a VS Code-based dynamic pricing system that uses advanced AI techniques to optimize e-commerce pricing in real-time. The system considers various factors such as demand patterns, customer segments, competitor pricing, contextual factors (weather, events), and applies reinforcement learning to maximize revenue while maintaining customer satisfaction.

## Key Features

- **Demand Forecasting**: ML/DL models to predict future product demand
- **Price Elasticity Modeling**: Determine how price changes affect demand
- **RL-based Optimal Pricing**: Reinforcement learning agents for price optimization
- **Customer Segmentation**: Tailored pricing for different customer segments
- **Competitor Price Scraping**: Monitor and react to competitor pricing
- **Psychological Pricing**: Implement psychological price points (₹499 vs ₹500)
- **Context-Aware Pricing**: Factor in weather, events, and other contextual data
- **Explainable AI**: SHAP-based explanations for pricing decisions
- **Real-time Dashboard**: Streamlit app for monitoring and adjusting pricing
- **API Backend**: FastAPI for model inference and pricing recommendations

## Project Structure

```
smartdynamic/
├── data/                  # Datasets
├── notebooks/             # Jupyter notebooks for modeling & EDA
├── src/                   # Python modules
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── demand_forecasting.py
│   ├── reinforcement_agent.py
│   ├── multiarmed_bandit.py
│   ├── price_optimizer.py
│   └── shap_explainer.py
├── api/                   # FastAPI backend
│   ├── main.py
│   └── routes.py
├── app/                   # Streamlit frontend
│   └── app.py
├── requirements.txt       # Python dependencies
├── README.md
└── Dockerfile (optional)
```

## Tech Stack

- **Python, Pandas, Scikit-learn, XGBoost, Prophet, LSTM (Keras)**
- **Stable-Baselines3** (Reinforcement Learning)
- **SHAP, LIME** (Explainability)
- **Streamlit** (Frontend UI)
- **FastAPI** (Backend API)
- **PostgreSQL/MongoDB** (Storage)
- **Docker** (for containerization)
- **VS Code** with extensions: Python, Jupyter, Pylance, GitLens

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smartdynamic.git
cd smartdynamic
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file with necessary API keys and configurations
cp .env.example .env
# Edit .env with your API keys and settings
```

## Usage

### Running the API

```bash
cd api
uvicorn main:app --reload
```

The API will be available at http://localhost:8000. API documentation is available at http://localhost:8000/docs.

### Running the Streamlit App

```bash
cd app
streamlit run app.py
```

The Streamlit app will be available at http://localhost:8501.

## Development

1. Data preparation and EDA are in the `notebooks/` directory
2. Core logic is implemented in the `src/` modules
3. API endpoints are defined in `api/routes.py`
4. Frontend components are in `app/app.py`

## Milestones

- Week 1: Project setup and tools installation
- Week 2: Data ingestion and ETL pipeline
- Week 3: Forecasting model development
- Week 4: Reinforcement learning agent & MAB testing
- Week 5: Psychological & context-aware pricing logic
- Week 6: SHAP explainability integration
- Week 7: Streamlit + FastAPI apps development
- Week 8: Testing, documentation, Dockerization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Your Name Vaibhav Chauhan
