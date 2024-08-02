import time
import yfinance as yf
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import alpaca_trade_api as tradapi
from utilities import generate_signals, trading_strategy, optimize_strategy, review_performance, \
    check_buying_power, calculate_shares, check_order_status, execute_trade, monitor_portfolio# Alpaca API setup


# Initialize the key, secret key, and url for the API
APCA_API_KEY_ID = 'PKQXTH7EXBAOF3KTE3C2'
APCA_API_SECRET_KEY = 'X5G2afLIH3ENlyBfIPAaPOu94oNUApg6RSgqVAhE'
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'

# Store api in a variable
api = tradapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL, api_version='v2')


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
