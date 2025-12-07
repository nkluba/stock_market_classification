import yfinance as yf
from fredapi import Fred
import pandas as pd
import argparse
import os
from datetime import datetime
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Configure root logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Silence yfinance internal logs
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)

# Load FRED API key
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)


# Fetch price history and dividends for a ticker
def load_market_data_from_yahoo(symbol, start_date, end_date):
    try:
        ticker_obj = yf.Ticker(symbol)
        price_data = ticker_obj.history(start=start_date, end=end_date)

        dividend_data = ticker_obj.dividends.to_frame().reset_index()
        dividend_data.columns = ["Date", "Dividends"]
        dividend_data["Ticker"] = symbol

        price_data["Ticker"] = symbol

        return price_data.reset_index(), dividend_data

    except Exception as exc:
        logging.error(f"Failed to retrieve Yahoo Finance data for {symbol}: {exc}")
        return pd.DataFrame(), pd.DataFrame()


# Fetch earnings calendar for a ticker
def load_earnings_calendar_from_yahoo(symbol, limit=8):
    try:
        ticker_obj = yf.Ticker(symbol)
        earnings_data = ticker_obj.get_earnings_dates(limit=limit)

        if earnings_data is None or earnings_data.empty:
            logging.warning(f"No earnings data available for {symbol}")
            return pd.DataFrame()

        earnings_data = earnings_data.rename_axis("Date").reset_index()
        earnings_data["Ticker"] = symbol

        return earnings_data

    except Exception as exc:
        logging.error(f"Failed to retrieve Yahoo earnings data for {symbol}: {exc}")
        return pd.DataFrame()


# Fetch macroeconomic series from FRED
def load_fred_time_series(series_id, start_date, end_date):
    try:
        series_data = fred.get_series(series_id, start_date, end_date)
        return series_data

    except Exception as exc:
        logging.error(f"Failed to retrieve FRED series {series_id}: {exc}")
        return pd.Series()


# Validate stock price data
def validate_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Date", "Close", "Ticker"}

    if stock_data.empty:
        logging.warning("Stock data is empty")
        return stock_data

    missing_cols = required_columns.difference(stock_data.columns)
    if missing_cols:
        logging.error(f"Stock data missing columns: {missing_cols}")
        return pd.DataFrame()

    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data = stock_data.dropna(subset=["Date", "Close", "Ticker"])
    stock_data = stock_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return stock_data


# Validate dividends data
def validate_dividends_data(dividends_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Date", "Dividends", "Ticker"}

    if dividends_data.empty:
        return dividends_data

    missing_cols = required_columns.difference(dividends_data.columns)
    if missing_cols:
        logging.error(f"Dividends data missing columns: {missing_cols}")
        return pd.DataFrame()

    dividends_data["Date"] = pd.to_datetime(dividends_data["Date"])
    dividends_data["Dividends"] = pd.to_numeric(dividends_data["Dividends"], errors="coerce")

    dividends_data = dividends_data.dropna(subset=["Date", "Dividends", "Ticker"])
    dividends_data = dividends_data[dividends_data["Dividends"] >= 0]
    dividends_data = dividends_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return dividends_data


# Validate earnings data
def validate_earnings_data(earnings_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Date", "Reported EPS", "Ticker"}

    if earnings_data.empty:
        return earnings_data

    missing_cols = required_columns.difference(earnings_data.columns)
    if missing_cols:
        logging.error(f"Earnings data missing columns: {missing_cols}")
        return pd.DataFrame()

    earnings_data["Date"] = pd.to_datetime(earnings_data["Date"])
    earnings_data["Reported EPS"] = pd.to_numeric(
        earnings_data["Reported EPS"], errors="coerce"
    )

    earnings_data = earnings_data.dropna(subset=["Date", "Reported EPS", "Ticker"])
    earnings_data = earnings_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return earnings_data


# Validate macroeconomic time series
def validate_macro_data(macro_data: pd.Series) -> pd.Series:
    if macro_data.empty:
        return macro_data

    macro_data = macro_data.dropna()
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data = macro_data[macro_data.index <= pd.Timestamp.today()]

    return macro_data


# Collect and merge stock, dividends, and earnings data
def collect_and_merge_market_data(symbols, start_date, end_date):
    stock_frames = []
    dividend_frames = []
    earnings_frames = []

    for symbol in symbols:
        price_data, dividend_data = load_market_data_from_yahoo(symbol, start_date, end_date)

        if not price_data.empty:
            stock_frames.append(price_data)

        if not dividend_data.empty:
            dividend_frames.append(dividend_data)

        earnings_data = load_earnings_calendar_from_yahoo(symbol)
        if not earnings_data.empty:
            earnings_frames.append(earnings_data)

    stock_merged = (
        pd.concat(stock_frames, ignore_index=True)
        if stock_frames else pd.DataFrame()
    )

    dividends_merged = (
        pd.concat(dividend_frames, ignore_index=True)
        if dividend_frames else pd.DataFrame()
    )

    earnings_merged = (
        pd.concat(earnings_frames, ignore_index=True)
        if earnings_frames else pd.DataFrame()
    )

    return stock_merged, dividends_merged, earnings_merged


# Main data ingestion pipeline
def run_backtest_ingestion_pipeline():
    symbols = filter_active_tickers(get_tickers_list())

    fred_series_id = "DTB3"
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    stock_data, dividends_data, earnings_data = collect_and_merge_market_data(
        symbols, start_date, end_date
    )

    stock_data = validate_stock_data(stock_data)
    dividends_data = validate_dividends_data(dividends_data)
    earnings_data = validate_earnings_data(earnings_data)

    macro_series = load_fred_time_series(fred_series_id, start_date, end_date)
    macro_data = validate_macro_data(macro_series)

    os.makedirs("data_exports/market", exist_ok=True)
    os.makedirs("data_exports/macro", exist_ok=True)

    if not stock_data.empty:
        stock_data.to_csv("data_exports/market/equity_prices.csv", index=False)

    if not dividends_data.empty:
        dividends_data.to_csv("data_exports/market/equity_dividends.csv", index=False)

    if not earnings_data.empty:
        earnings_data.to_csv("data_exports/market/equity_earnings.csv", index=False)

    if not macro_data.empty:
        macro_data.to_csv(
            "data_exports/macro/interest_rates.csv",
            header=True,
            index=True
        )

    return stock_data, dividends_data, earnings_data, macro_data


# Compute annualized EPS growth
def calculate_growth_rate(earnings_df):
    try:
        earnings_df["Reported EPS"] = earnings_df["Reported EPS"].astype(float)
        earnings_df["Quarterly Growth"] = earnings_df["Reported EPS"].pct_change()
        growth_rates = earnings_df["Quarterly Growth"].dropna()

        if growth_rates.empty:
            return None

        return np.mean(growth_rates) * 4
    except Exception as e:
        logging.error(f"Error calculating growth rate: {e}")
        return None


# Filter only actively traded tickers
def filter_active_tickers(tickers):
    active_tickers = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info

            if info is None or info.get("lastPrice") is None:
                continue

            active_tickers.append(ticker)

        except Exception:
            continue

    logging.info(f"Active tickers after filtering: {active_tickers}")
    return active_tickers


def get_sp500_tickers():
    sources = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    ]

    for url in sources:
        try:
            if url.endswith(".csv"):
                df = pd.read_csv(url)
                tickers = df["Symbol"].astype(str).str.upper().tolist()
            else:
                tables = pd.read_html(url)
                df = tables[0]
                tickers = df["Symbol"].astype(str).str.upper().tolist()

            if tickers:
                return tickers
        except Exception as exc:
            logging.warning("Ticker source failed %s %s", url, exc)

    return []


# get tickers list
def get_tickers_list(default=None):
    try:
        tickers = get_sp500_tickers()
        if tickers:
            logging.info("Loaded %d tickers from S&P 500 public dataset", len(tickers))
            return tickers

        if default is None:
            default = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        logging.warning(
            "All remote ticker sources failed using default list %s",
            ",".join(default),
        )
        return default

    except Exception as exc:
        logging.error("Error while building tickers list %s", exc)
        return []


# Build classification feature matrix
def build_classification_dataset(stock_data, dividends_data, earnings_data):
    stock_grouped = stock_data.groupby("Ticker")

    first_price = stock_grouped["Close"].first()
    last_price = stock_grouped["Close"].last()

    # Future return used only for target
    future_return = (last_price - first_price) / first_price

    # Volatility feature
    volatility = stock_grouped["Close"].apply(lambda x: x.pct_change().std())

    # Dividend yield feature
    if not dividends_data.empty:
        total_dividends = dividends_data.groupby("Ticker")["Dividends"].sum()
        dividend_yield = total_dividends / first_price
    else:
        dividend_yield = pd.Series(0.0, index=first_price.index)

    # EPS growth feature
    eps_growth = {}

    for ticker in first_price.index:
        ticker_earnings = earnings_data[earnings_data["Ticker"] == ticker]
        if ticker_earnings.empty:
            eps_growth[ticker] = 0.0
        else:
            growth = calculate_growth_rate(ticker_earnings.copy())
            eps_growth[ticker] = growth if growth is not None else 0.0

    eps_growth = pd.Series(eps_growth, name="EPSGrowth")

    # Feature matrix without leakage
    features = pd.DataFrame(
        {
            "Volatility": volatility,
            "DividendYield": dividend_yield,
            "EPSGrowth": eps_growth,
        }
    )

    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    # Predictive target based on future returns
    future_return = future_return.loc[features.index]
    median_future_return = future_return.median()
    target = (future_return >= median_future_return).astype(int)
    target.name = "OutperformFuture"

    return features, target


# Train stock classifier
def train_stock_classifier(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    os.makedirs("data_exports", exist_ok=True)
    joblib.dump(model, "data_exports/stock_classifier.pkl")

    logging.info(f"ROC AUC: {roc_auc}")
    logging.info(classification_report(y_test, y_pred))

    return model


def run_single_symbol_inference(symbol, start_date="2023-08-01"):
    model_path = "data_exports/models/equity_classifier.pkl"

    if not os.path.exists(model_path):
        logging.error("Trained model not found. Run training first.")
        return

    model = joblib.load(model_path)
    end_date = datetime.today().strftime("%Y-%m-%d")

    stock_data, dividends_data = load_market_data_from_yahoo(
        symbol, start_date, end_date
    )
    earnings_data = load_earnings_calendar_from_yahoo(symbol)

    stock_data = validate_stock_data(stock_data)
    dividends_data = validate_dividends_data(dividends_data)
    earnings_data = validate_earnings_data(earnings_data)

    if stock_data.empty:
        logging.error(f"No usable stock data for {symbol}")
        return

    returns = stock_data["Close"].pct_change().dropna()
    volatility = returns.std()

    if not dividends_data.empty and "Dividends" in dividends_data.columns:
        first_price = stock_data["Close"].iloc[0]
        dividend_yield = dividends_data["Dividends"].sum() / first_price
    else:
        dividend_yield = 0.0

    if not earnings_data.empty:
        eps_growth = calculate_growth_rate(earnings_data.copy())
        eps_growth = eps_growth if eps_growth is not None else 0.0
    else:
        eps_growth = 0.0

    features = pd.DataFrame(
        [[volatility, dividend_yield, eps_growth]],
        columns=["Volatility", "DividendYield", "EPSGrowth"]
    )

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    logging.info(f"Single symbol inference for {symbol}")
    logging.info(f"Volatility: {volatility}")
    logging.info(f"DividendYield: {dividend_yield}")
    logging.info(f"EPSGrowth: {eps_growth}")
    logging.info(f"Predicted OutperformFuture: {int(prediction)}")
    logging.info(f"Predicted Probability: {probability:.4f}")


# classification pipeline
def run_equity_classification_pipeline():
    stock_data, dividends_data, earnings_data, macro_data = (
        run_backtest_ingestion_pipeline()
    )

    features, target = build_classification_dataset(
        stock_data,
        dividends_data,
        earnings_data
    )

    os.makedirs("data_exports/models", exist_ok=True)

    train_stock_classifier(features, target)

    logging.info("Equity classification pipeline completed")


def run_sp500_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--ticker", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "train":
        run_equity_classification_pipeline()

    elif args.mode == "predict":
        if args.ticker is None:
            logging.error("Ticker must be provided in predict mode")
            return

        run_single_symbol_inference(args.ticker)