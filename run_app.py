"""
Entry point script for running the SmartDynamic pricing app.
"""

import os
import sys
import streamlit.web.cli as stcli
from dotenv import load_dotenv

def run_app():
    """Run the Streamlit app."""
    # Load environment variables
    load_dotenv()
    
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Path to app.py
    app_path = os.path.join(dir_path, "app", "app.py")
    
    # Set API URL environment variable if not already set
    if not os.getenv("API_BASE_URL"):
        os.environ["API_BASE_URL"] = "http://localhost:8000"
    
    # Run Streamlit app
    sys.argv = [
        "streamlit",  
        "run", 
        app_path,
        "--server.port", os.getenv("STREAMLIT_PORT", "8501"),
        "--browser.serverAddress", "localhost",
        "--server.headless", "true"
    ]
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_app()
