import requests
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import alpaca_trade_api as tradapi
import time

# Alpaca API setup
APCA_API_KEY_ID = 'PKQXTH7EXBAOF3KTE3C2'
APCA_API_SECRET_KEY = 'X5G2afLIH3ENlyBfIPAaPOu94oNUApg6RSgqVAhE'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')


# Generate signals
def generate_signals(df, short_window, long_window):
    df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
    df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1.0, np.where(df['Short_MA'] < df['Long_MA'], -1.0, 0.0))
    df['Position'] = df['Signal'].diff()
    return df


# Trading strategy implementation
def trading_strategy(df):
    short_window = 40
    long_window = 100
    initial_capital = 1000000.0
    risk_per_trade = 0.01  # 1% of capital
    stop_loss_pct = 0.02  # 2% stop-loss
    take_profit_pct = 0.04  # 4% take-profit

    df = generate_signals(df, short_window, long_window)

    portfolio = pd.DataFrame(index=df.index)
    portfolio['Cash'] = initial_capital
    portfolio['Position'] = 0
    portfolio['Holdings'] = 0.0
    portfolio['Total'] = initial_capital
    portfolio['Buy_Price'] = 0.0
    portfolio['Stop_Loss'] = 0.0
    portfolio['Take_Profit'] = 0.0

    def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
        ps = (capital * risk_per_trade) / stop_loss_pct
        if ps < 1:
            return 0
        return ps

    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1.0:
            position_size = calculate_position_size(portfolio['Cash'].iloc[i - 1], risk_per_trade, stop_loss_pct)
            shares = int(position_size // df['Close'].iloc[i])
            if shares > 0:
                portfolio.loc[df.index[i], 'Position'] = shares
                portfolio.loc[df.index[i], 'Buy_Price'] = df['Close'].iloc[i]
                portfolio.loc[df.index[i], 'Stop_Loss'] = df['Close'].iloc[i] * (1 - stop_loss_pct)
                portfolio.loc[df.index[i], 'Take_Profit'] = df['Close'].iloc[i] * (1 + take_profit_pct)
                portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] - (shares * df['Close'].iloc[i])
            else:
                portfolio.loc[df.index[i], 'Position'] = 0
        elif df['Position'].iloc[i] == -1.0 or (portfolio['Position'].iloc[i - 1] > 0 and
                                                (df['Close'].iloc[i] <= portfolio['Stop_Loss'].iloc[i - 1] or
                                                 df['Close'].iloc[i] >= portfolio['Take_Profit'].iloc[i - 1])):
            portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1] + (
                    portfolio['Position'].iloc[i - 1] * df['Close'].iloc[i])
            portfolio.loc[df.index[i], 'Position'] = 0
            portfolio.loc[df.index[i], 'Buy_Price'] = 0
            portfolio.loc[df.index[i], 'Stop_Loss'] = 0
            portfolio.loc[df.index[i], 'Take_Profit'] = 0
        else:
            portfolio.loc[df.index[i], 'Position'] = portfolio['Position'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Cash'] = portfolio['Cash'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Buy_Price'] = portfolio['Buy_Price'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Stop_Loss'] = portfolio['Stop_Loss'].iloc[i - 1]
            portfolio.loc[df.index[i], 'Take_Profit'] = portfolio['Take_Profit'].iloc[i - 1]

        portfolio.loc[df.index[i], 'Holdings'] = portfolio['Position'].iloc[i] * df['Close'].iloc[i]
        portfolio.loc[df.index[i], 'Total'] = portfolio['Cash'].iloc[i] + portfolio['Holdings'].iloc[i]

    return portfolio


# Optimize strategy
def optimize_strategy(df):
    short_windows = [20, 40, 60]
    long_windows = [100, 150, 200]
    best_result = None
    best_params = None

    for short_window, long_window in product(short_windows, long_windows):
        df = generate_signals(df, short_window, long_window)
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
    try:
        account = api.get_account()
        available_cash = float(account.cash)
        return available_cash
    except Exception as e:
        print(f"Failed to get account information: {e}")
        return 0.0


# Calculate shares
def calculate_shares(buying_power, price_per_share, risk_per_trade):
    max_shares = int(buying_power / price_per_share)
    shares_to_trade = int(risk_per_trade * max_shares)
    return max(1, shares_to_trade)


# Check order status
def check_order_status(order_id, api):
    try:
        order = api.get_order(order_id)
        print(f"Order status: {order.status}")
        return order.status
    except Exception as e:
        print(f"Failed to get order status {e}")
        return None


# Execute trade
def execute_trade(signal, symbol, shares, api):
    if shares <= 0:
        print(f"Invalid quantity ({shares}) for order. No trade executed")
        return

    available_cash = check_buying_power(api)

    try:
        price_per_share = api.get_latest_trade(symbol).price
        cost_of_trade = price_per_share * shares

        if signal == 1:  # Buy signal
            if available_cash >= cost_of_trade:
                order = api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Buy order executed: {shares} shares of {symbol} at {price_per_share}")

                # Check order status until filled or timeout
                start_time = time.time()
                while True:
                    order_status = check_order_status(order.id, api)
                    if order_status == 'filled':
                        print(f"Order filled: {shares} shares of {symbol}")
                        break
                    elif time.time() - start_time > 60:  # Timeout after 60 seconds
                        print(f"Order not filled after 60 seconds. Status: {order_status}")
                        break
                    time.sleep(5)  # Check status every 5 seconds
            else:
                print(f"Insufficient buying power for buy order. Available cash: {available_cash:.2f}, Required: ${cost_of_trade:.2f}")

        elif signal == -1:  # Sell signal
            time.sleep(10)  # Delay to ensure previous order is processed
            try:
                position = api.get_position(symbol)
                if int(position.qty) >= shares:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=shares,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    print(f"Sell order executed: {shares} shares of {symbol} at {price_per_share}")

                    # Check order status until filled or timeout
                    start_time = time.time()
                    while True:
                        order_status = check_order_status(order.id, api)
                        if order_status == 'filled':
                            print(f"Order filled: {shares} shares of {symbol}")
                            break
                        elif time.time() - start_time > 60:  # Timeout after 60 seconds
                            print(f"Order not filled after 60 seconds. Status: {order_status}")
                            break
                        time.sleep(5)  # Check status every 5 seconds
                else:
                    print(f"Not enough shares to sell. Available shares: {position.qty}, Required: {shares}")
            except Exception as e:
                print(f"No position found for {symbol}: {e}")
    except Exception as e:
        print(f"Failed to submit order: {e}")


# Monitor portfolio
def monitor_portfolio(symbol):
    try:
        position = api.get_position(symbol)
        print(f"Current position in {symbol}: {position.qty} shares")
        print(f"Market value: {position.market_value}")
        print(f"Cost basis: {position.cost_basis}")
        print(f"Unrealized P/L: {position.unrealized_pl}")
    except Exception as e:
        print(f"No position found for {symbol}: {e}")

    # Add more logic for monitoring and adjusting the portfolio as needed


def main():
    symbol = 'AAPL'
    data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

    optimized_params = optimize_strategy(data)
    new_data = generate_signals(data, *optimized_params)
    portfolio = trading_strategy(new_data)
    review_performance(portfolio)

    cash = check_buying_power(api)
    print(f"Buying power: {cash}")
    trade = api.get_latest_trade(symbol)
    latest_price = trade.price
    print(f"current price: {latest_price}")
    shares = calculate_shares(cash, latest_price, 0.01)
    execute_trade(1, symbol, shares, api)
    time.sleep(60)
    execute_trade(-1, symbol, shares, api)
    monitor_portfolio(symbol)


if __name__ == '__main__':
    main()
