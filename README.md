# 1. Create virtual environment
python -m venv weather_app_env

# 2. Activate it
weather_app_env\Scripts\activate

# 3. Install packages
pip install streamlit pandas matplotlib seaborn scikit-learn plotly

# 4. Run the app
streamlit run weather_app.py
