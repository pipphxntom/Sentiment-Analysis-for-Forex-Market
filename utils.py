import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import io

def generate_sample_data():
    """
    Generate sample data for demonstration purposes.
    
    Returns:
        tuple: (sample_news_df, sample_forex_df)
    """
    # Generate sample news data
    sample_news = generate_sample_news()
    
    # Generate sample forex data
    sample_forex = generate_sample_forex(sample_news['date'].min(), sample_news['date'].max())
    
    return sample_news, sample_forex

def generate_sample_news(start_date=None, end_date=None, num_samples=200):
    """
    Generate sample news data for demonstration.
    
    Args:
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        num_samples (int): Number of news samples to generate
        
    Returns:
        pd.DataFrame: Sample news data
    """
    # Default date range if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert to datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate random dates within range
    date_range = (end_dt - start_dt).days
    random_dates = [start_dt + timedelta(days=random.randint(0, date_range)) for _ in range(num_samples)]
    random_dates.sort()
    
    # Sample news headlines and content
    bullish_templates = [
        "Dollar strengthens against major currencies on positive economic data",
        "Euro surges after ECB hints at maintaining rates",
        "Pound sterling rallies on Brexit trade deal hopes",
        "Yen weakens as risk appetite improves in global markets",
        "Australian dollar climbs on strong employment figures",
        "Canadian dollar reaches multi-year high on rising oil prices",
        "Currency markets bullish as global growth outlook improves",
        "Dollar index shows upward momentum amid treasury yield rise",
        "Federal Reserve comments boost USD trading sentiment",
        "Forex traders optimistic as economic indicators show recovery"
    ]
    
    bearish_templates = [
        "Dollar weakens as Fed signals continued monetary stimulus",
        "Euro falls to six-month low against dollar on economic concerns",
        "Pound plummets on new Brexit uncertainties and trade tensions",
        "Yen strengthens as investors seek safe-haven currencies",
        "Australian dollar declines as central bank cuts growth forecast",
        "Canadian dollar drops on falling oil prices and economic data",
        "Currency markets show bearish signals amid global slowdown fears",
        "Dollar index tumbles to lowest level in two years",
        "Traders bearish on USD following disappointing jobs report",
        "Forex volatility increases as recession fears grow"
    ]
    
    neutral_templates = [
        "Currency markets show mixed signals amid conflicting economic data",
        "Dollar trades sideways as traders await Federal Reserve meeting",
        "Euro holds steady against dollar ahead of ECB announcement",
        "Pound shows limited movement as Brexit negotiations continue",
        "Yen trades in narrow range amid balanced risk assessment",
        "Australian dollar moves sideways on neutral RBA statement",
        "Canadian dollar unchanged following Bank of Canada decision",
        "Forex traders cautious as conflicting market signals emerge",
        "Dollar index relatively unchanged in quiet trading session",
        "Currency pairs consolidate as markets assess economic outlook"
    ]
    
    # Generate news articles with sentiment bias
    news_data = []
    sentiment_distribution = ['bullish'] * 35 + ['bearish'] * 35 + ['neutral'] * 30
    
    for i in range(num_samples):
        # Select sentiment bias for this article
        sentiment = random.choice(sentiment_distribution)
        
        # Select template based on sentiment
        if sentiment == 'bullish':
            template = random.choice(bullish_templates)
            # Add more bullish content
            additional_content = "Analysts expect further gains in the coming weeks. Economic indicators show positive momentum, with employment and manufacturing data exceeding expectations. Traders are increasingly taking long positions."
        elif sentiment == 'bearish':
            template = random.choice(bearish_templates)
            # Add more bearish content
            additional_content = "Market analysts warn of further declines ahead. Economic data has been disappointing, with key indicators missing forecasts. Trading sentiment remains cautious with increased short positions."
        else:  # neutral
            template = random.choice(neutral_templates)
            # Add more neutral content
            additional_content = "Analysts remain divided on the future direction. While some economic indicators are positive, others show potential weakness. Traders are maintaining balanced positions as they await clearer signals."
        
        # Create full text by combining template and additional content
        full_text = f"{template} {additional_content}"
        
        # Add some randomness to the date (avoid having all news articles on the same dates)
        date = random_dates[i].strftime('%Y-%m-%d')
        
        news_data.append({
            'date': date,
            'text': full_text
        })
    
    return pd.DataFrame(news_data)

def generate_sample_forex(start_date, end_date, currency_pair='EUR/USD'):
    """
    Generate sample forex OHLC data for demonstration.
    
    Args:
        start_date (str): Start date
        end_date (str): End date
        currency_pair (str): Currency pair symbol
        
    Returns:
        pd.DataFrame: Sample forex data
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate date range (business days only)
    dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    # Base parameters for the simulated price
    if currency_pair == 'EUR/USD':
        base_price = 1.15
        daily_volatility = 0.005
    elif currency_pair == 'GBP/USD':
        base_price = 1.35
        daily_volatility = 0.006
    elif currency_pair == 'USD/JPY':
        base_price = 110.0
        daily_volatility = 0.5
    else:
        base_price = 1.0
        daily_volatility = 0.005
    
    # Generate random walk prices
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, daily_volatility, len(dates))
    price_path = base_price * (1 + np.cumsum(returns))
    
    # Add some trend to make it more realistic
    trend = np.linspace(0, 0.1, len(dates))
    price_path = price_path + trend
    
    # Generate OHLC data
    forex_data = []
    for i, date in enumerate(dates):
        close_price = price_path[i]
        daily_range = close_price * daily_volatility * random.uniform(0.5, 2.0)
        
        high_price = close_price + daily_range / 2
        low_price = close_price - daily_range / 2
        open_price = low_price + random.uniform(0, daily_range)
        
        forex_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'volume': int(random.uniform(50000, 200000))
        })
    
    return pd.DataFrame(forex_data)

def load_example_news():
    """
    Load example news data for the application.
    
    Returns:
        pd.DataFrame: Example news data
    """
    # Generate some realistic example news data
    example_data = generate_sample_news(
        start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        num_samples=300
    )
    
    return example_data

def load_example_forex():
    """
    Load example forex data for the application.
    
    Returns:
        pd.DataFrame: Example forex data
    """
    # Generate some realistic example forex data
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    example_data = generate_sample_forex(start_date, end_date, 'EUR/USD')
    
    return example_data
