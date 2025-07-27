# Forex Market Sentiment Analysis Tool

## Overview
This application analyzes sentiment in forex news and social media data to provide insights for trading decisions. Using Natural Language Processing (NLP), it classifies sentiment as bullish, bearish, or neutral, and provides backtesting capabilities to evaluate trading strategies based on sentiment signals.

## Features
- **Sentiment Analysis**: Analyze news/social media content using NLP models
- **Interactive Dashboard**: View sentiment trends and distributions
- **Backtesting**: Test trading strategies based on sentiment signals
- **Visualization**: Track sentiment trends over time with interactive charts
- **Data Upload**: Import your own news and forex historical data
- **Example Data**: Use built-in example data for demonstrations

## Technologies Used
- **Streamlit**: Interactive web application framework
- **NLTK**: Natural Language Toolkit for text processing and analysis
- **pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **scikit-learn**: Machine learning components

## How to Use
1. **Data Upload**: Upload news/social media data and forex historical data
2. **Sentiment Analysis**: Run analysis to classify sentiment
3. **View Results**: Explore sentiment distributions and trends
4. **Backtesting**: Test trading strategies based on sentiment signals

## Data Format Requirements
- **News Data**: CSV file with columns 'date' (YYYY-MM-DD format) and 'text'
- **Forex Data**: CSV file with columns 'date', 'open', 'high', 'low', 'close'

## Installation and Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/forex-sentiment-analyzer.git
    cd forex-sentiment-analyzer
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -e .
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

    The application will be available at `http://localhost:8501`.
