import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def backtest_strategy(
    forex_data,
    sentiment_data,
    strategy="Simple Sentiment",
    timeframe="Daily",
    sentiment_threshold=0.3,
    position_size=0.1,
    initial_capital=10000
):
    """
    Backtest a trading strategy based on sentiment analysis.
    
    Args:
        forex_data (pd.DataFrame): DataFrame with forex price data (OHLC)
        sentiment_data (pd.DataFrame): DataFrame with sentiment analysis results
        strategy (str): Strategy type to backtest
        timeframe (str): Timeframe for the backtest ('Daily' or 'Weekly')
        sentiment_threshold (float): Threshold for sentiment score to trigger trades
        position_size (float): Portion of capital to risk per trade (0-1)
        initial_capital (float): Initial capital for the backtest
        
    Returns:
        dict: Dictionary containing backtest results
    """
    # Ensure the indices are datetime format
    forex_data = forex_data.copy()
    if not isinstance(forex_data.index, pd.DatetimeIndex):
        forex_data.index = pd.to_datetime(forex_data.index)
    
    sentiment_data = sentiment_data.copy()
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    
    # Aggregate sentiment by date
    daily_sentiment = sentiment_data.groupby('date').agg({
        'sentiment': lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutral',
        'score': 'mean'
    })
    
    # Resample if weekly timeframe is selected
    if timeframe == "Weekly":
        # Resample forex data to weekly
        forex_weekly = forex_data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        # Resample sentiment data to weekly
        # For weekly sentiment, take the most common sentiment of the week
        sentiment_weekly = daily_sentiment.resample('W').agg({
            'sentiment': lambda x: x.mode().iloc[0] if not x.mode().empty else 'neutral',
            'score': 'mean'
        })
        
        forex_data = forex_weekly
        daily_sentiment = sentiment_weekly
    
    # Initialize variables for backtest
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    position_price = 0
    trades = []
    equity_curve = []
    dates = []
    
    # Generate signals based on selected strategy
    signals = generate_signals(forex_data, daily_sentiment, strategy, sentiment_threshold)
    
    # Run backtest
    for date, row in signals.iterrows():
        current_price = row['close']
        
        # Record equity for this day
        equity_curve.append(capital + position * (current_price - position_price) * (initial_capital * position_size / position_price if position_price else 0))
        dates.append(date)
        
        # Process buy signal
        if row['signal'] == 1 and position <= 0:  # Buy signal and not already long
            # Close any existing short position
            if position == -1:
                profit = position_price - current_price
                capital += profit * (initial_capital * position_size / position_price)
                trades.append({
                    'date': date,
                    'action': 'sell',  # Closing short = selling
                    'price': current_price,
                    'profit_pct': (profit / position_price) * 100,
                    'capital': capital
                })
            
            # Open new long position
            position = 1
            position_price = current_price
            trades.append({
                'date': date,
                'action': 'buy',
                'price': current_price,
                'capital': capital
            })
            
        # Process sell signal
        elif row['signal'] == -1 and position >= 0:  # Sell signal and not already short
            # Close any existing long position
            if position == 1:
                profit = current_price - position_price
                capital += profit * (initial_capital * position_size / position_price)
                trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': current_price,
                    'profit_pct': (profit / position_price) * 100,
                    'capital': capital
                })
            
            # Open new short position
            position = -1
            position_price = current_price
            trades.append({
                'date': date,
                'action': 'buy',  # Opening short = buying to sell later
                'price': current_price,
                'capital': capital
            })
    
    # Close any open position at the end of the backtest
    if position != 0:
        last_date = signals.index[-1]
        last_price = signals.iloc[-1]['close']
        
        if position == 1:  # Long position
            profit = last_price - position_price
            capital += profit * (initial_capital * position_size / position_price)
            trades.append({
                'date': last_date,
                'action': 'sell',
                'price': last_price,
                'profit_pct': (profit / position_price) * 100,
                'capital': capital
            })
        else:  # Short position
            profit = position_price - last_price
            capital += profit * (initial_capital * position_size / position_price)
            trades.append({
                'date': last_date,
                'action': 'sell',  # Closing short
                'price': last_price,
                'profit_pct': (profit / position_price) * 100,
                'capital': capital
            })
    
    # Create equity curve DataFrame
    equity_df = pd.DataFrame({
        'date': dates,
        'equity': equity_curve
    })
    equity_df.set_index('date', inplace=True)
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_df, trades_df, initial_capital)
    
    # Calculate monthly returns
    monthly_returns = calculate_monthly_returns(equity_df)
    
    return {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics,
        'monthly_returns': monthly_returns
    }

def generate_signals(forex_data, sentiment_data, strategy, threshold):
    """
    Generate trading signals based on sentiment data and chosen strategy.
    
    Args:
        forex_data (pd.DataFrame): Forex price data
        sentiment_data (pd.DataFrame): Sentiment analysis data
        strategy (str): Trading strategy type
        threshold (float): Sentiment threshold for signals
        
    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    # Merge forex data with sentiment data
    signals = forex_data.copy()
    signals = signals.join(sentiment_data, how='left')
    
    # Fill missing sentiment values
    signals['sentiment'] = signals['sentiment'].fillna('neutral')
    signals['score'] = signals['score'].fillna(0.5)
    
    # Initialize signal column (0 = no signal, 1 = buy, -1 = sell)
    signals['signal'] = 0
    
    if strategy == "Simple Sentiment":
        # Generate signals based on simple sentiment values
        signals.loc[signals['sentiment'] == 'bullish', 'signal'] = 1
        signals.loc[signals['sentiment'] == 'bearish', 'signal'] = -1
        
    elif strategy == "Sentiment Momentum":
        # Calculate the rolling average sentiment score (5-day window)
        sentiment_numeric = signals['sentiment'].map({'bullish': 1, 'bearish': -1, 'neutral': 0})
        signals['sentiment_ma5'] = sentiment_numeric.rolling(window=5).mean().fillna(0)
        
        # Generate signals based on sentiment momentum
        signals.loc[signals['sentiment_ma5'] > threshold, 'signal'] = 1
        signals.loc[signals['sentiment_ma5'] < -threshold, 'signal'] = -1
        
    elif strategy == "Sentiment Reversal":
        # Calculate the difference between current and previous day sentiment
        sentiment_numeric = signals['sentiment'].map({'bullish': 1, 'bearish': -1, 'neutral': 0})
        signals['sentiment_diff'] = sentiment_numeric - sentiment_numeric.shift(1)
        
        # Generate signals based on sentiment reversals
        signals.loc[(signals['sentiment'] == 'bearish') & (signals['sentiment_diff'] < 0), 'signal'] = 1
        signals.loc[(signals['sentiment'] == 'bullish') & (signals['sentiment_diff'] > 0), 'signal'] = -1
    
    # Forward fill signals to ensure they persist until a new signal
    signals['signal'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return signals

def calculate_performance_metrics(equity_df, trades_df, initial_capital):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        equity_df (pd.DataFrame): Equity curve dataframe
        trades_df (pd.DataFrame): Trades dataframe
        initial_capital (float): Initial capital amount
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Calculate total return
    final_equity = equity_df['equity'].iloc[-1]
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    daily_returns = equity_df['equity'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
    
    # Calculate max drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()
    
    # Calculate win rate
    if 'profit_pct' in trades_df.columns:
        profitable_trades = trades_df[trades_df['profit_pct'] > 0]
        win_rate = (len(profitable_trades) / len(trades_df[trades_df['action'] == 'sell'])) * 100 if len(trades_df[trades_df['action'] == 'sell']) > 0 else 0
    else:
        win_rate = 0
    
    return {
        'total_return': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades_df[trades_df['action'] == 'buy'])
    }

def calculate_monthly_returns(equity_df):
    """
    Calculate monthly returns from equity curve.
    
    Args:
        equity_df (pd.DataFrame): Equity curve dataframe
        
    Returns:
        pd.DataFrame: DataFrame with monthly returns
    """
    # Resample equity curve to month-end and calculate returns
    monthly_equity = equity_df['equity'].resample('M').last()
    monthly_returns = monthly_equity.pct_change().dropna() * 100
    
    # Format as a DataFrame with year and month
    monthly_returns_df = pd.DataFrame(monthly_returns)
    monthly_returns_df.columns = ['return']
    
    # Extract year and month
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    # Pivot to get year as rows and month as columns
    pivot_table = monthly_returns_df.pivot_table(
        values='return',
        index='year',
        columns='month',
        aggfunc='sum'
    )
    
    # Rename month columns
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    pivot_table.rename(columns=month_names, inplace=True)
    
    # Add annual return column
    pivot_table['Annual'] = pivot_table.sum(axis=1)
    
    return pivot_table.round(2)
