# SmartDynamic Pricing - Streamlit Frontend

This directory contains the Streamlit frontend for the SmartDynamic Pricing system. The frontend provides a user-friendly interface to interact with the pricing API and visualize pricing recommendations.

## Features

- **Price Recommendations**: Get AI-powered pricing recommendations with explanations
- **Product Insights**: Analyze product performance, sales history, and elasticity
- **Market Analysis**: Compare your prices with market trends and competitors
- **Settings**: Configure pricing strategies and system preferences

## Directory Structure

```
app/
├── app.py                # Main Streamlit app
├── api_client.py         # API client for backend interaction
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── visualizations.py     # Data visualization functions
└── static/
    └── style.css         # Custom CSS styles
```

## Running the App

You can run the Streamlit app directly using:

```bash
streamlit run app/app.py
```

Or use the provided entry point script:

```bash
python run_app.py
```

## API Integration

The app connects to the SmartDynamic Pricing API to fetch data and recommendations. The API base URL can be configured through environment variables:

```bash
export API_BASE_URL=http://localhost:8000
```

## Customization

- The app's appearance can be customized by modifying the CSS in `static/style.css`
- System-wide configuration settings are available in `config.py`

## Development

To set up a development environment:

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app in development mode:
```bash
streamlit run app/app.py --server.runOnSave=true
```

## Mock Data

The app includes mock data generators in `utils.py` for development and demonstration purposes. These are used automatically if the API is unavailable.
