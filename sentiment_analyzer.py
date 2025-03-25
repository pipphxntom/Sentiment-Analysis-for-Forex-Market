import re
import string
import pandas as pd
import numpy as np
from datetime import datetime

# Import NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK resources
# Download all required resources to ensure they're available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

# Forex-specific sentiment lexicon
# This extends the default VADER lexicon with forex-specific terms
forex_lexicon = {
    # Bullish terms
    'bull': 2.0,
    'bullish': 2.0,
    'rally': 1.5,
    'surge': 1.5,
    'soar': 1.5,
    'gain': 1.0,
    'rise': 1.0,
    'climbing': 1.0,
    'uptrend': 1.5,
    'strengthen': 1.0,
    'breakout': 1.5,
    'support': 0.5,
    'buy': 1.0,
    'buying': 1.0,
    'higher': 0.8,
    'peak': 0.7,
    'boom': 1.2,
    'recovery': 1.0,
    'outperform': 1.0,
    'momentum': 0.6,
    
    # Bearish terms
    'bear': -2.0,
    'bearish': -2.0,
    'slump': -1.5,
    'decline': -1.5,
    'crash': -2.0,
    'fall': -1.0,
    'drop': -1.0,
    'plunge': -1.8,
    'plummet': -1.8,
    'tumble': -1.5,
    'downtrend': -1.5,
    'weaken': -1.0,
    'breakdown': -1.5,
    'resistance': -0.5,
    'sell': -1.0,
    'selling': -1.0,
    'lower': -0.8,
    'bottom': -0.5,
    'recession': -1.8,
    'underperform': -1.0,
    
    # Forex specific
    'dovish': -1.0,
    'hawkish': 1.0,
    'inflation': -0.5,
    'deflation': -1.0,
    'rate hike': -0.8,
    'rate cut': 0.8,
    'easing': 0.7,
    'tightening': -0.7,
    'intervention': -0.3,
    'deficit': -0.8,
    'surplus': 0.8,
    'liquidity': 0.5,
    'volatility': -0.3,
    'stimulus': 0.8,
    'austerity': -0.8,
    'gdp growth': 1.0,
    'gdp contraction': -1.0,
    'unemployment': -0.8,
    'employment': 0.8,
    'central bank': 0.0,  # Neutral by itself
    'parity': 0.0,        # Neutral by itself
    'reserve': 0.3,
    'debt': -0.7,
}

# Update VADER lexicon with forex terms
sid.lexicon.update(forex_lexicon)

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation,
    tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Simple tokenization by splitting on whitespace instead of using word_tokenize
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back to text
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text

def analyze_sentiment(text, model_type="NLTK VADER"):
    """
    Analyze sentiment of a text and classify as bullish, bearish, or neutral.
    
    Args:
        text (str): Preprocessed text to analyze
        model_type (str): Type of model to use ('NLTK VADER' or 'Spacy TextBlob')
        
    Returns:
        dict: Dictionary containing sentiment label and score
    """
    if not text:
        return {"label": "neutral", "score": 0.5}
    
    if model_type == "NLTK VADER":
        # Get sentiment scores using VADER
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        # Classify sentiment based on compound score
        if compound_score >= 0.05:
            label = "bullish"
            score = (compound_score + 1) / 2  # Scale to 0-1 range
        elif compound_score <= -0.05:
            label = "bearish"
            score = 1 - (compound_score + 1) / 2  # Scale to 0-1 range
        else:
            label = "neutral"
            score = 0.5
            
    elif model_type == "Spacy TextBlob":
        # Import TextBlob if not imported yet
        from textblob import TextBlob
        
        # Get sentiment using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Classify sentiment based on polarity
        if polarity >= 0.1:
            label = "bullish"
            score = (polarity + 1) / 2  # Scale to 0-1 range
        elif polarity <= -0.1:
            label = "bearish"
            score = 1 - (polarity + 1) / 2  # Scale to 0-1 range
        else:
            label = "neutral"
            score = 0.5
            
    else:
        # Default to neutral if model type is not recognized
        label = "neutral"
        score = 0.5
        
    return {"label": label, "score": score}

def analyze_news_sentiment(news_df, model_type="NLTK VADER"):
    """
    Analyze sentiment for a dataframe of news items.
    
    Args:
        news_df (pandas.DataFrame): DataFrame containing 'text' column with news content
        model_type (str): Type of model to use for sentiment analysis
        
    Returns:
        pandas.DataFrame: Original DataFrame with sentiment analysis results added
    """
    # Create a copy to avoid modifying the original
    df = news_df.copy()
    
    # Ensure the DataFrame has a 'text' column
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    # Preprocess text and analyze sentiment
    df['preprocessed_text'] = df['text'].apply(preprocess_text)
    
    # Apply sentiment analysis to each row
    sentiments = df['preprocessed_text'].apply(lambda x: analyze_sentiment(x, model_type))
    
    # Extract sentiment labels and scores
    df['sentiment'] = sentiments.apply(lambda x: x['label'])
    df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
    
    return df

def aggregate_daily_sentiment(sentiment_df):
    """
    Aggregate sentiment by date to get daily sentiment trends.
    
    Args:
        sentiment_df (pandas.DataFrame): DataFrame with 'date' and 'sentiment' columns
        
    Returns:
        pandas.DataFrame: DataFrame with daily sentiment counts and percentages
    """
    # Make sure date column is datetime
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Group by date and sentiment
    daily_sentiment = sentiment_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    
    # Calculate total articles per day
    daily_sentiment['total'] = daily_sentiment.sum(axis=1)
    
    # Calculate percentages
    for sentiment in ['bullish', 'bearish', 'neutral']:
        if sentiment in daily_sentiment.columns:
            daily_sentiment[f'{sentiment}_pct'] = (daily_sentiment[sentiment] / daily_sentiment['total']) * 100
    
    # Calculate bullish-bearish ratio
    if 'bullish' in daily_sentiment.columns and 'bearish' in daily_sentiment.columns:
        daily_sentiment['bull_bear_ratio'] = daily_sentiment['bullish'] / (daily_sentiment['bearish'] + 0.001)  # Avoid division by zero
    
    # Calculate net sentiment (bullish - bearish)
    if 'bullish' in daily_sentiment.columns and 'bearish' in daily_sentiment.columns:
        daily_sentiment['net_sentiment'] = daily_sentiment['bullish'] - daily_sentiment['bearish']
    
    return daily_sentiment
