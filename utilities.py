import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import alpaca_trade_api as tradapi
import time
import logging


def generate_signals(df, short_window, long_window):
    """
    Generate trading signals based on moving averages.
    :param df:
    :param short_window:
    :param long_window:
    :return: data frame
    """
    try:
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")

        df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        df['Signal'] = np.where(df['Short_MA'] > df['Long_MA'], 1.0, np.where(df['Short_MA'] < df['Long_MA'], -1.0, 0.0))
        df['Position'] = df['Signal'].diff()
        return df
    except ValueError as e:
        logging.error(f"ValueError in generate_signals: {e}")
        return df
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return df


def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
    """
    Calculate the position size based on capital, risk per trade, and stop-loss percentage
    :param capital:
    :param risk_per_trade:
    :param stop_loss_pct:
    :return: position size or 0
    """
    try:
        if capital <= 0 or risk_per_trade <= 0 or stop_loss_pct <= 0:
            raise ValueError("Capital, risk per trade, and stop loss percentage must be possitive values")

        ps = (capital * risk_per_trade) / stop_loss_pct
        return max(0, ps)
    except ValueError as e:
        logging.error(f"ValueError in calculate_position_size: {e}")
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        return 0


def trading_strategy(df, initial_capital=1000000.0, risk_per_trade=0.01, stop_loss_pct=0.02, take_profit_pct=0.04):
    """
    Implement the trading strategy and simulate its performance
    :param df:
    :param initial_capital: 1000000
    :param risk_per_trade: 0.01
    :param stop_loss_pct: 0.02
    :param take_profit_pct: 0.04
    :return: portfolio
    """
    try:
        df = generate_signals(df, short_window=40, long_window=100)
        portfolio = pd.DataFrame(index=df.index)
        portfolio['Cash'] = initial_capital
        portfolio['Position'] = 0
        portfolio['Holdings'] = 0.0
        portfolio['Total'] = initial_capital
        portfolio['Buy_Price'] = 0.0
        portfolio['Stop_Loss'] = 0.0
        portfolio['Take_Profit'] = 0.0

        for i in range(1, len(df)):
            try:
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
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")

        return portfolio
    except ValueError as e:
        logging.error(f"ValueError in trading_strategy: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error in trading strategy: {e}")
        return pd.DataFrame()


def optimize_strategy(df):
    """
    Optimize the trading strategy by finding the best moving average window sizes.
    :param df:
    :return: best parameters
    """
    try:
        short_windows = [20, 40, 60]
        long_windows = [100, 150, 200]
        best_result = None
        best_params = None

        for short_window, long_window in product(short_windows, long_windows):
            try:
                df = generate_signals(df, short_window, long_window)
                if df is None:
                    raise ValueError(f"Failed to generate signals for short_window={short_window}, long_window={long_window}")

                portfolio = trading_strategy(df)
                if portfolio.empty:
                    raise ValueError(f"Failed to execute trading strategy for short_window={short_window}, long_window={long_window}")

                final_value = portfolio['Total'].iloc[-1]

                if best_result is None or final_value > best_result:
                    best_result = final_value
                    best_params = (short_window, long_window)
            except ValueError as e:
                logging.error(f"ValueError in optimize_strategy with short_window={short_window}, long_window={long_window}: {e}")
            except Exception as e:
                logging.error(f"Error optimizing with short_window={short_window}, long_window={long_window}: {e}")

        if best_params is not None:
            logging.info(f"Best Parameters: Short Window: {best_params[0]}, Long Window: {best_params[1]}")
            logging.info(f"Best Final Portfolio Value: ${best_result:.2f}")
            return best_params
        else:
            raise ValueError("No valid parameters found during optimization")
    except ValueError as e:
        logging.error(f"ValueError in optimize_strategy: {e}")
        return None
    except Exception as e:
        logging.error(f"Error in optimizing strategy: {e}")
        return None


def review_performance(portfolio):
    """
    Review the performance of the trading strategy
    :param portfolio:
    :return:
    """
    try:
        if portfolio.empty:
            raise ValueError("Portfolio is empty")

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

        return portfolio
    except ValueError as e:
        logging.error(f"ValueError in review_performance: {e}")
    except Exception as e:
        logging.error(f"Error reviewing performance: {e}")


def check_buying_power(api, symbol):
    """
    Check the available buying power in the Alpaca account
    :param api:
    :return: available cash or 0.0
    """
    try:
        account = api.get_account()
        if account is None:
            raise ValueError("Failed to fetch account details")

        buying_power = float(account.cash)
        positions = api.get_position(symbol)
        if positions is None:
            raise ValueError(f"Failed to fetch position details for {symbol}")

        logging.info(f"Current buying power: ${buying_power:.2f}")
        logging.info(f"Current position in {symbol}: {positions.qty} shares")
        return buying_power, float(positions.qty)
    except ValueError as e:
        logging.error(f"ValueError in check_buying_power: {e}")
        return 0.0, 0.0
    except Exception as e:
        logging.error(f"Failed to get account information: {e}")
        return 0.0, 0.0


def calculate_shares(buying_power, price_per_share, risk_per_trade):
    """
    Calculate the number of shares to trade based on buying power, price per share, and risk per trade.
    :param buying_power:
    :param price_per_share:
    :param risk_per_trade:
    :return: number of shares
    """
    try:
        if not isinstance(buying_power, (int, float)) or not isinstance(price_per_share, (int, float)) or not isinstance(risk_per_trade, (int, float)):
            raise ValueError("Buying power, price per share, and risk per trade must be positive values")

        max_shares = int(buying_power / price_per_share)
        shares_to_trade = int(risk_per_trade * max_shares)
        return max(1, shares_to_trade)
    except Exception as e:
        logging.error(f"Error calculating shares: {e}")
        return 1


def check_order_status(order_id, api):
    """
    Check the status of an order
    :param order_id:
    :param api:
    :return: order status
    """
    try:
        order = api.get_order(order_id)
        logging.info(f"Order status: {order.status}")
        return order.status
    except Exception as e:
        logging.error(f"Failed to get order status {e}")
        return None


def execute_trade(signal, symbol, shares, api):
    """
    Execute a trade based on the trading signal
    :param signal:
    :param symbol:
    :param shares:
    :param api:
    :return:
    """
    if shares <= 0:
        logging.error(f"Invalid quantity ({shares}) for order. No trade executed")
        return

    buying_power, current_qty = check_buying_power(api, symbol)

    try:
        price_per_share = api.get_latest_trade(symbol).price
        cost_of_trade = price_per_share * shares

        if signal == 1:  # Buy signal
            if buying_power >= cost_of_trade:
                order = api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                logging.info(f"Buy order executed: {shares} shares of {symbol} at {price_per_share}")

                # Check order status until filled or timeout
                start_time = time.time()
                while True:
                    order_status = check_order_status(order.id, api)
                    if order_status == 'filled':
                        logging.info(f"Order filled: {shares} shares of {symbol}")
                        break
                    elif time.time() - start_time > 60:  # Timeout after 60 seconds
                        logging.error(f"Order not filled after 60 seconds. Status: {order_status}")
                        break
                    time.sleep(5)  # Check status every 5 seconds
            else:
                logging.error(f"Insufficient buying power for buy order. Available cash: {buying_power:.2f}, Required: ${cost_of_trade:.2f}")

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
                    logging.info(f"Sell order executed: {shares} shares of {symbol} at {price_per_share}")

                    # Check order status until filled or timeout
                    start_time = time.time()
                    while True:
                        order_status = check_order_status(order.id, api)
                        if order_status == 'filled':
                            logging.info(f"Order filled: {shares} shares of {symbol}")
                            break
                        elif time.time() - start_time > 60:  # Timeout after 60 seconds
                            logging.error(f"Order not filled after 60 seconds. Status: {order_status}")
                            break
                        time.sleep(5)  # Check status every 5 seconds
                else:
                    logging.error(f"Not enough shares to sell. Available shares: {position.qty}, Required: {shares}")
            except Exception as e:
                logging.error(f"No position found for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Failed to submit order: {e}")



def monitor_portfolio(symbol, api):
    """
    Monitor the portfolio for the given symbol
    :param symbol:
    :return:
    """
    try:
        position = api.get_position(symbol)
        logging.info(f"Current position in {symbol}: {position.qty} shares")
        logging.info(f"Market value: {position.market_value}")
        logging.info(f"Cost basis: {position.cost_basis}")
        logging.info(f"Unrealized P/L: {position.unrealized_pl}")
    except Exception as e:
        logging.error(f"No position found for {symbol}: {e}")


def manage_risk(portfolio):
    try:
        max_drawdown = portfolio['Drawdown'].min()
        if max_drawdown < -0.2:
            logging.warning(f"Maximum drawdown of {max_drawdown * 100:.2f}% exceeds acceptable limit of -20%.")
            return True
        return False
    except Exception as e:
        logging.error(f"Unexpected error in manage_risk: {e}")
        return False