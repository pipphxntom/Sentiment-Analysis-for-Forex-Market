import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import random

def process_news_data(news_df):
    """
    Process raw news data to prepare for sentiment analysis.
    
    Args:
        news_df (pd.DataFrame): DataFrame containing news data
        
    Returns:
        pd.DataFrame: Processed news data
    """
    # Create a copy to avoid modifying the original
    df = news_df.copy()
    
    # Check and ensure required columns exist
    required_cols = ['date', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in news data")
    
    # Convert date to datetime if not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with missing dates or text
    df = df.dropna(subset=['date', 'text'])
    
    # Ensure text is string type
    df['text'] = df['text'].astype(str)
    
    # Sort by date
    df = df.sort_values('date')
    
    # For news articles, extract only the content relevant to forex
    df['text'] = df['text'].apply(extract_forex_content)
    
    return df

def process_forex_data(forex_df):
    """
    Process raw forex price data.
    
    Args:
        forex_df (pd.DataFrame): DataFrame containing forex OHLC price data
        
    Returns:
        pd.DataFrame: Processed forex data
    """
    # Create a copy to avoid modifying the original
    df = forex_df.copy()
    
    # Check and ensure required columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in forex data")
    
    # Convert date to datetime if not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with missing required values
    df = df.dropna(subset=required_cols)
    
    # Ensure price columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_values('date')
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Add some technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to forex data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate moving averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def extract_forex_content(text):
    """
    Extract content relevant to forex from a news article.
    
    Args:
        text (str): Full text of a news article
        
    Returns:
        str: Extracted forex-relevant content
    """
    # If the text is short, return as is
    if len(text) < 200:
        return text
    
    # List of forex-related keywords
    forex_keywords = [
        'forex', 'fx', 'currency', 'currencies', 'exchange rate', 
        'dollar', 'euro', 'pound', 'yen', 'usd', 'eur', 'gbp', 'jpy',
        'central bank', 'interest rate', 'inflation', 'economic', 
        'fed', 'federal reserve', 'ecb', 'bank of england', 'boe',
        'bank of japan', 'boj', 'rba', 'reserve bank', 'monetary policy',
        'trade', 'gdp', 'growth', 'employment', 'unemployment', 
        'treasury', 'yield', 'bond', 'stimulus', 'recession',
        'bull', 'bear', 'bullish', 'bearish', 'market sentiment',
        'traders', 'trading', 'volatile', 'volatility'
    ]
    
    # Create regex pattern for forex keywords
    pattern = r'(?i)(\b' + r'\b|\b'.join(forex_keywords) + r'\b)'
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Select sentences containing forex keywords
    relevant_sentences = [s for s in sentences if re.search(pattern, s, re.IGNORECASE)]
    
    # If no relevant sentences found, return first and last paragraph
    if not relevant_sentences:
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            return paragraphs[0] + '\n\n' + paragraphs[-1]
        else:
            return text
    
    # Join relevant sentences
    return ' '.join(relevant_sentences)

def align_data_dates(news_df, forex_df):
    """
    Align news and forex data by date for analysis.
    
    Args:
        news_df (pd.DataFrame): Processed news data
        forex_df (pd.DataFrame): Processed forex data
        
    Returns:
        tuple: (aligned_news_df, aligned_forex_df)
    """
    # Create copies to avoid modifying originals
    news = news_df.copy()
    forex = forex_df.copy()
    
    # Ensure date column is datetime
    news['date'] = pd.to_datetime(news['date'])
    
    # Get the range of dates from both datasets
    news_dates = set(news['date'].dt.date)
    forex_dates = set(forex.index.date)
    
    # Find common dates
    common_dates = news_dates.intersection(forex_dates)
    
    # Filter both datasets to only include common dates
    news = news[news['date'].dt.date.isin(common_dates)]
    forex = forex[forex.index.date.isin(common_dates)]
    
    return news, forex
