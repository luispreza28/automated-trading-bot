import requests
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import alpaca_trade_api as tradapi
import time

# Alpaca API setup
APCA_API_KEY_ID = 'PKTUOK4XOTI6RMRC87WT'
APCA_API_SECRET_KEY = 'x9mfNEJtVS2eAwLd1vzSsrfJWHwqfN0Ujs3KQDai'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')


# Function to fetch data from an API if needed
def fetch_data(api_url):
    # Use api url to retrieve a response
    response = requests.get(api_url)

    # Convert response into a json dictionary object and store it into a variable
    data = response.json()

    # Return data as a DataFrame
    df = pd.DataFrame(data)
    return df


# Trading strategy implementation
def trading_strategy(df):
    # Parameters
    short_window = 40
    long_window = 100
    initial_capital = 100000.0
    risk_per_trade = 0.01  # 1% of capital
    stop_loss_pct = 0.02  # 2% stop-loss
    take_profit_pct = 0.04  # 4% take-profit

    # Calculating moving averages
    df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

    # Generate signals
    df['Signal'] = 0.0
    df.iloc[short_window:, df.columns.get_loc('Signal')] = np.where(
        df.iloc[short_window:, df.columns.get_loc('Short_MA')] > df.iloc[short_window:, df.columns.get_loc('Long_MA')],
        1.0,
        0.0
    )
    # Initialize portfolio
    portfolio = pd.DataFrame(index=df.index)
    portfolio['Cash'] = initial_capital
    portfolio['Position'] = 0
    portfolio['Holdings'] = 0.0
    portfolio['Total'] = initial_capital
    portfolio['Buy_Price'] = 0.0
    portfolio['Stop_Loss'] = 0.0
    portfolio['Take_Profit'] = 0.0

    # Calculate position size
    def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
        ps = (capital * risk_per_trade) / stop_loss_pct
        if ps < 1:
            return 0
        return ps

    # Simulate trading with risk management
    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1.0:  # Buy Signal
            position_size = calculate_position_size(portfolio['Cash'].iloc[i - 1], risk_per_trade, stop_loss_pct)
            shares = int(position_size // df['Close'].iloc[i])
            if shares > 0: # Ensure shares is positive
                portfolio.loc[df.index[i], 'Position'] = shares
                portfolio.loc[df.index[i], 'Buy_Price'] = df['Close'].iloc[i]
                portfolio.loc[df.index[i], 'Stop_Loss'] = df['Close'].iloc[i] * (1 - stop_loss_pct)
                portfolio.loc[df.index[i], 'Take_Profit'] = df['Close'].iloc[i] * (1 + take_profit_pct)
                portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] - (shares * df['Close'].iloc[i])
                # print(f"BUY: Date: {df.index[i]}, Shares: {shares}, Buy Price: {df['Close'].iloc[i]}, "
                #       f"Cash after buy: {portfolio['Cash'].iloc[i]}, Total: {portfolio['Total'].iloc[i]}")
            else:
                portfolio.loc[df.index[i], 'Position'] = 0
        elif df['Position'].iloc[i] == -1.0 or (portfolio['Position'].iloc[i - 1] > 0 and
                                                (df['Close'].iloc[i] <= portfolio['Stop_Loss'].iloc[i - 1] or
                                                 df['Close'].iloc[i] >= portfolio['Take_Profit'].iloc[i - 1])):  # Sell Signal or Stop-Loss/Take-Profit
            portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] + (
                    portfolio['Position'].iloc[i - 1] * df['Close'].iloc[i])
            portfolio.loc[df.index[i], 'Position'] = 0
            portfolio.loc[df.index[i], 'Buy_Price'] = 0
            portfolio.loc[df.index[i], 'Stop_Loss'] = 0
            portfolio.loc[df.index[i], 'Take_Profit'] = 0
            # print(f"SELL: Date: {df.index[i]}, Sell Price: {df['Close'].iloc[i]}, "
            #       f"Cash after sell: {portfolio['Cash'].iloc[i]}, Total: {portfolio['Total'].iloc[i]}")
        else:  # No Signal
            portfolio.loc[df.index[i], 'Position'] = portfolio['Position'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Buy_Price'] = portfolio['Buy_Price'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Stop_Loss'] = portfolio['Stop_Loss'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Take_Profit'] = portfolio['Take_Profit'].iloc[i - 1]

        portfolio.loc[df.index[i], 'Holdings'] = portfolio['Position'].iloc[i] * df['Close'].iloc[i]
        portfolio.loc[df.index[i], 'Total'] = portfolio['Cash'].iloc[i] + portfolio['Holdings'].iloc[i]

        # Print intermediate results for verification
        # print(f"Date: {df.index[i]}, Signal: {df['Signal'].iloc[i]}, Position: {portfolio['Position'].iloc[i]}, "
        #       f"Cash: {portfolio['Cash'].iloc[i]}, Holdings: {portfolio['Holdings'].iloc[i]}, "
        #       f"Total: {portfolio['Total'].iloc[i]}, Stop_Loss: {portfolio['Stop_Loss'].iloc[i]}, "
        #       f"Take_Profit: {portfolio['Take_Profit'].iloc[i]}")

    return portfolio


# Optimize strategy
def optimize_strategy(df):
    short_windows = [20, 40, 60]
    long_windows = [100, 150, 200]
    best_result = None
    best_params = None

    for short_window, long_window in product(short_windows, long_windows):
        df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        df['Signal'] = 0.0
        df.iloc[short_window:, df.columns.get_loc('Signal')] = np.where(
            df.iloc[short_window:, df.columns.get_loc('Short_MA')] > df.iloc[short_window:, df.columns.get_loc('Long_MA')],
            1.0,
            0.0
        )
        df['Position'] = df['Signal'].diff()
        portfolio = trading_strategy(df)

        final_value = portfolio['Total'].iloc[-1]
        if best_result is None or final_value > best_result:
            best_result = final_value
            best_params = (short_window, long_window)

    print(f"Best Parameters: Short Window: {best_params[0]}, Long Window: {best_params[1]}")
    print(f"Best Final Portfolio Value: ${best_result:.2f}")
    return best_params


# Review performance
def review_performance(portfolio):
    portfolio['Returns'] = portfolio['Total'].pct_change()
    portfolio['Cumulative Returns'] = (1 + portfolio['Returns']).cumprod()
    portfolio['Drawdown'] = portfolio['Total'] / portfolio['Total'].cummax() - 1

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(portfolio['Total'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(portfolio['Cumulative Returns'], label='Cumulative Returns')
    plt.plot(portfolio['Drawdown'], label='Drawdown')
    plt.title('Cumulative Returns and Drawdown')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    print(f"Final Portfolio Value: ${portfolio['Total'].iloc[-1]:.2f}")
    print(f"Max Drawdown: {portfolio['Drawdown'].min() * 100:.2f}%")
    print(f"Total Returns: {portfolio['Cumulative Returns'].iloc[-1] * 100:.2f}%")


# Check buying power
def check_buying_power(api):
    account = api.get_account()
    available_cash = float(account.cash)
    return available_cash


# Execute trade
def execute_trade(signal, symbol, shares, api):
    # Check if the qunatity is valid
    if shares <= 0:
        print(f"Invalid quantity ({shares}) for order. No trade executed")
        return

    # Check buying power before executing trade
    available_cash = check_buying_power(api)

    # Get the current price of the symbol
    price_per_share = api.get_latest_trade(symbol).price
    cost_of_trade = price_per_share * shares

    if signal == 1: # Buy signal
        if available_cash >= cost_of_trade:
            api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"Buy order Executed: {shares} shares of {symbol} at {price_per_share}")
        else:
            print(f"Insufficient buying power for buy order. Available cash: {available_cash:.2f}, Required: ${cost_of_trade:.2f}")

    elif signal == -1: # Sell signal
        api.submit_order(
            symbol=symbol,
            qty=shares,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Sell Order Executed: {shares} shares of {symbol} at {price_per_share}")


# Risk management
def risk_management(portfolio, max_drawdown_limit=0.20):
    current_drawdown = portfolio['Drawdown'].iloc[-1]
    if current_drawdown <= -max_drawdown_limit:
        print(f"Max drawdown limit reached: {current_drawdown:.2%}")
        print("Stopping trade now...")


# Monitoring
def monitor_portfolio(portfolio, interval=60):
    while True:
        print(f"Current portfolio Value: ${portfolio['Total'].iloc[-1]:.2f}")
        print(f"Current Drawdown: {portfolio['Drawdown'].iloc[-1] * 100:.2f}%")
        print(f"Current Returns: {portfolio['Cumulative Returns'].iloc[-1] * 100:.2f}%")
        risk_management(portfolio)
        time.sleep(interval)


def main():
    symbol = 'AAPL'
    df = yf.download(symbol, start='2020-01-01', end='2023-01-01')

    # Uncomment the following line if you want to fetch data from an API
    # df = fetch_data('https://example.com/data')

    # Optimize strategy and get the best parameters
    best_params = optimize_strategy(df)
    short_window, long_window = best_params

    # Apply the best parameters to the trading strategy
    df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    df['Signal'] = 0.0
    df.iloc[short_window:, df.columns.get_loc('Signal')] = np.where(
        df.iloc[short_window:, df.columns.get_loc('Short_MA')] > df.iloc[short_window:, df.columns.get_loc('Long_MA')],
        1.0, 0.0
    )
    df['Position'] = df['Signal'].diff()

    # Run trading strategy with the best parameters
    portfolio = trading_strategy(df)

    # Print intermediate results for verification
    print(df.head())
    print(portfolio.head())

    # Plot portfolio value and review performance
    review_performance(portfolio)

    # Execute trade
    for i in range(len(df)):
        if df['Position'].iloc[i] == 1.0:
            execute_trade(1, symbol, int(portfolio['Position'].iloc[i]), api)
        elif df['Position'].iloc[i] == -1.0:
            execute_trade(-1, symbol, int(portfolio['Position'].iloc[i]), api)

    # Monitor portfolio
    monitor_portfolio(portfolio)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted. Exiting...")
