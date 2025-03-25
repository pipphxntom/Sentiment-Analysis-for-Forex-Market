import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Import custom modules
from sentiment_analyzer import analyze_sentiment, preprocess_text
from backtester import backtest_strategy
from data_processor import process_news_data, process_forex_data
from utils import generate_sample_data, load_example_news, load_example_forex

# Set page config
st.set_page_config(
    page_title="Forex Market Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'news_data' not in st.session_state:
    st.session_state.news_data = None
if 'forex_data' not in st.session_state:
    st.session_state.forex_data = None
if 'sentiment_results' not in st.session_state:
    st.session_state.sentiment_results = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Title and introduction
st.title("Forex Market Sentiment Analysis Tool")
st.write("""
This tool analyzes news and social media data to determine market sentiment for forex trading.
Using Natural Language Processing (NLP), it classifies sentiment as bullish, bearish, or neutral,
providing valuable insights for trading decisions.
""")

# Create tabs for different functionality
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Sentiment Analysis", "Backtesting", "About"])

with tab1:
    st.header("Upload Data")
    
    st.subheader("News/Social Media Data")
    news_upload_method = st.radio(
        "Choose news data source:",
        ["Upload CSV", "Use Example Data"],
        key="news_upload_method"
    )
    
    if news_upload_method == "Upload CSV":
        news_file = st.file_uploader("Upload news/social media data (CSV format)", type="csv")
        st.info("""
        Expected format: CSV with columns 'date' (YYYY-MM-DD format) and 'text' (news content).
        """)
        
        if news_file is not None:
            try:
                news_data = pd.read_csv(news_file)
                if 'date' in news_data.columns and 'text' in news_data.columns:
                    st.session_state.news_data = news_data
                    st.success(f"Successfully loaded news data with {len(news_data)} entries")
                else:
                    st.error("CSV must contain 'date' and 'text' columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        if st.button("Load Example News Data"):
            st.session_state.news_data = load_example_news()
            st.success(f"Loaded example news data with {len(st.session_state.news_data)} entries")
    
    st.subheader("Forex Historical Data")
    forex_upload_method = st.radio(
        "Choose forex data source:",
        ["Upload CSV", "Use Example Data"],
        key="forex_upload_method"
    )
    
    if forex_upload_method == "Upload CSV":
        forex_file = st.file_uploader("Upload forex historical data (CSV format)", type="csv")
        st.info("""
        Expected format: CSV with columns 'date', 'open', 'high', 'low', 'close'.
        Date should be in YYYY-MM-DD format.
        """)
        
        if forex_file is not None:
            try:
                forex_data = pd.read_csv(forex_file)
                required_cols = ['date', 'open', 'high', 'low', 'close']
                if all(col in forex_data.columns for col in required_cols):
                    st.session_state.forex_data = forex_data
                    st.success(f"Successfully loaded forex data with {len(forex_data)} entries")
                else:
                    st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        if st.button("Load Example Forex Data"):
            st.session_state.forex_data = load_example_forex()
            st.success(f"Loaded example forex data with {len(st.session_state.forex_data)} entries")
    
    if st.session_state.news_data is not None:
        st.subheader("News Data Preview")
        st.dataframe(st.session_state.news_data.head())
    
    if st.session_state.forex_data is not None:
        st.subheader("Forex Data Preview")
        st.dataframe(st.session_state.forex_data.head())

with tab2:
    st.header("Sentiment Analysis")
    
    if st.session_state.news_data is None:
        st.warning("Please upload or load news data in the 'Data Upload' tab first.")
    else:
        st.write("Run sentiment analysis on the uploaded news data.")
        
        with st.form("sentiment_form"):
            st.write("Sentiment Analysis Parameters")
            model_type = st.selectbox(
                "Select NLP Model",
                ["NLTK VADER", "Spacy TextBlob"],
                index=0
            )
            
            date_range = st.date_input(
                "Select Date Range",
                value=(
                    datetime.strptime(st.session_state.news_data['date'].min(), '%Y-%m-%d').date(),
                    datetime.strptime(st.session_state.news_data['date'].max(), '%Y-%m-%d').date()
                ),
                min_value=datetime.strptime(st.session_state.news_data['date'].min(), '%Y-%m-%d').date(),
                max_value=datetime.strptime(st.session_state.news_data['date'].max(), '%Y-%m-%d').date()
            )
            
            submit_button = st.form_submit_button("Run Analysis")
        
        if submit_button or st.session_state.sentiment_results is not None:
            with st.spinner("Analyzing sentiment..."):
                # Filter news data based on date range if needed
                filtered_data = st.session_state.news_data
                if date_range:
                    start_date, end_date = date_range
                    filtered_data = filtered_data[
                        (pd.to_datetime(filtered_data['date']) >= pd.Timestamp(start_date)) &
                        (pd.to_datetime(filtered_data['date']) <= pd.Timestamp(end_date))
                    ]
                
                # Run sentiment analysis
                if submit_button or st.session_state.sentiment_results is None:
                    sentiment_results = []
                    for _, row in filtered_data.iterrows():
                        preprocessed_text = preprocess_text(row['text'])
                        sentiment = analyze_sentiment(preprocessed_text, model_type)
                        sentiment_results.append({
                            'date': row['date'],
                            'text': row['text'],
                            'sentiment': sentiment['label'],
                            'score': sentiment['score']
                        })
                    
                    st.session_state.sentiment_results = pd.DataFrame(sentiment_results)
            
            # Display results
            st.subheader("Sentiment Analysis Results")
            
            # Display summary
            sentiment_counts = st.session_state.sentiment_results['sentiment'].value_counts()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("Sentiment Distribution:")
                st.dataframe(pd.DataFrame({
                    'Sentiment': sentiment_counts.index,
                    'Count': sentiment_counts.values,
                    'Percentage': (sentiment_counts.values / len(st.session_state.sentiment_results) * 100).round(2)
                }))
            
            with col2:
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'bullish': 'green',
                        'bearish': 'red',
                        'neutral': 'gray'
                    }
                )
                st.plotly_chart(fig)
            
            # Display sentiment over time
            st.subheader("Sentiment Trend Over Time")
            sentiment_by_date = st.session_state.sentiment_results.groupby('date')['sentiment'].value_counts().unstack().fillna(0)
            
            # Normalize to percentage
            sentiment_percentage = sentiment_by_date.div(sentiment_by_date.sum(axis=1), axis=0) * 100
            
            fig = px.line(
                sentiment_percentage,
                x=sentiment_percentage.index,
                y=sentiment_percentage.columns,
                title="Daily Sentiment Distribution (%)",
                labels={'value': 'Percentage', 'variable': 'Sentiment', 'date': 'Date'},
                color_discrete_map={
                    'bullish': 'green',
                    'bearish': 'red',
                    'neutral': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table of sentiment results
            st.subheader("Detailed Sentiment Results")
            st.dataframe(st.session_state.sentiment_results)
            
            # Option to download results
            csv = st.session_state.sentiment_results.to_csv(index=False)
            st.download_button(
                label="Download Sentiment Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Backtesting")
    
    if st.session_state.sentiment_results is None or st.session_state.forex_data is None:
        st.warning("Please complete sentiment analysis and upload forex data before backtesting.")
    else:
        st.write("Backtest trading strategies based on sentiment analysis.")
        
        with st.form("backtest_form"):
            st.write("Backtesting Parameters")
            
            strategy = st.selectbox(
                "Select Trading Strategy",
                ["Simple Sentiment", "Sentiment Momentum", "Sentiment Reversal"],
                index=0
            )
            
            timeframe = st.selectbox(
                "Select Timeframe",
                ["Daily", "Weekly"],
                index=0
            )
            
            sentiment_threshold = st.slider(
                "Sentiment Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum sentiment score to consider for trading decisions."
            )
            
            position_size = st.slider(
                "Position Size (%)",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help="Percentage of capital to risk per trade."
            )
            
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            submit_button = st.form_submit_button("Run Backtest")
        
        if submit_button or st.session_state.backtest_results is not None:
            with st.spinner("Running backtest..."):
                # Prepare data for backtesting
                if submit_button or st.session_state.backtest_results is None:
                    # Ensure forex data has datetime index
                    forex_df = st.session_state.forex_data.copy()
                    forex_df['date'] = pd.to_datetime(forex_df['date'])
                    forex_df.set_index('date', inplace=True)
                    
                    # Ensure sentiment results have datetime
                    sentiment_df = st.session_state.sentiment_results.copy()
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                    
                    # Run backtest
                    backtest_results = backtest_strategy(
                        forex_df,
                        sentiment_df,
                        strategy=strategy,
                        timeframe=timeframe,
                        sentiment_threshold=sentiment_threshold,
                        position_size=position_size/100,
                        initial_capital=initial_capital
                    )
                    
                    st.session_state.backtest_results = backtest_results
            
            # Display backtest results
            st.subheader("Backtest Results")
            
            # Performance metrics
            metrics = st.session_state.backtest_results['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            
            # Equity curve
            st.subheader("Equity Curve")
            equity_df = st.session_state.backtest_results['equity_curve']
            
            fig = px.line(
                equity_df,
                x=equity_df.index,
                y='equity',
                title="Equity Curve",
                labels={'equity': 'Equity', 'index': 'Date'}
            )
            
            # Add buy and sell markers
            trades = st.session_state.backtest_results['trades']
            
            buy_points = trades[trades['action'] == 'buy']
            sell_points = trades[trades['action'] == 'sell']
            
            if not buy_points.empty:
                fig.add_scatter(
                    x=buy_points['date'],
                    y=buy_points['price'],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Buy'
                )
            
            if not sell_points.empty:
                fig.add_scatter(
                    x=sell_points['date'],
                    y=sell_points['price'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Sell'
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade details
            st.subheader("Trade History")
            st.dataframe(trades)
            
            # Monthly returns
            st.subheader("Monthly Returns (%)")
            monthly_returns = st.session_state.backtest_results['monthly_returns']
            st.dataframe(monthly_returns)
            
            # Download backtest results
            col1, col2 = st.columns(2)
            
            with col1:
                csv_trades = trades.to_csv(index=False)
                st.download_button(
                    label="Download Trade History",
                    data=csv_trades,
                    file_name="backtest_trades.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_equity = equity_df.reset_index().to_csv(index=False)
                st.download_button(
                    label="Download Equity Curve",
                    data=csv_equity,
                    file_name="backtest_equity.csv",
                    mime="text/csv"
                )

with tab4:
    st.header("About")
    
    st.subheader("Forex Market Sentiment Analysis Tool")
    st.write("""
    This tool combines natural language processing (NLP) with forex market data to provide
    sentiment-based trading insights. By analyzing news and social media content, the system
    classifies market sentiment as bullish, bearish, or neutral, potentially enhancing trading decisions.
    """)
    
    st.subheader("Features")
    st.markdown("""
    - **Sentiment Analysis**: Uses NLP to classify news/social media text sentiment
    - **Interactive Visualizations**: Track sentiment trends over time
    - **Backtesting**: Test trading strategies based on sentiment signals
    - **Data Upload**: Analyze your own news and forex data
    """)
    
    st.subheader("Methodology")
    st.markdown("""
    The sentiment analysis employs natural language processing techniques:
    
    1. **Text Preprocessing**: Tokenization, removing stopwords, lemmatization
    2. **Sentiment Classification**: Using lexicon-based models (VADER) or machine learning
    3. **Scoring**: Calculating sentiment scores for classification
    4. **Backtesting**: Simulating trades based on sentiment signals
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    - NLP models may not capture complex market nuances
    - Past performance doesn't guarantee future results
    - Sentiment is just one factor in forex market movements
    - Model accuracy depends on quality and relevance of input data
    """)
    
    st.subheader("Credits")
    st.markdown("Created by Shwetank Pandey")

# Footer
st.markdown("---")
st.markdown("© 2023 Forex Sentiment Analyzer | Created by Shwetank Pandey")
