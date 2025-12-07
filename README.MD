# Stock Market Classification Pipeline

Automated financial data ingestion and machine learning pipeline for **predicting future relative stock performance** using Yahoo Finance and FRED macroeconomic data.

The pipeline builds a clean dataset, filters delisted tickers, engineers financial features, and trains a **classification model** to predict whether a stock will **outperform the market median in the next period**.

This project includes **end-to-end ML engineering**, using steps of data validation, feature engineering, modeling, evaluation, and reproducible Docker deployment.

---

## ML Objective

**Problem formulation**

> Given historical price behavior, dividend yield, and earnings growth, predict whether a stock will **outperform the median future return**.

---

## Pipeline Overview

1. Loads S&P 500 ticker list from CSV.
2. Filters out delisted and inactive securities using live Yahoo Finance metadata.
3. Downloads historical prices and dividends from Yahoo Finance.
4. Retrieves earnings calendars and reported EPS.
5. Fetches macroeconomic interest rate data from FRED.
6. Applies validation and cleaning:
   * Schema checks
   * NaN handling
   * Non-positive dividend filtering
7. Engineers stock-level predictive features:
   * Volatility (risk)
   * Dividend yield
   * EPS growth
8. Constructs a **future-return based binary classification target**.
9. Trains a **RandomForest classifier**.
10. Evaluates performance using **ROC AUC, precision, recall, and F1-score**.
11. Saves all datasets and trained model locally.

---

## Data Collected

### Stock Price Data

* Date
* Open
* High
* Low
* Close
* Volume
* Ticker

### Dividend Data

* Date
* Dividend amount
* Ticker

### Earnings Data

* Date
* EPS estimate
* Reported EPS
* EPS surprise
* Ticker

### Macroeconomic Data

* 10-Year US Treasury Yield (FRED: GS10)

---

## Output Structure

```
backtest_data/
├── stock_data.csv
├── dividends_data.csv
├── earnings_data.csv
├── macro_data.csv
└── stock_classifier.pkl
```

---

## Run

```bash
docker build -t stock-classifier .

docker run --rm \
  -e FRED_API_KEY=your_fred_api_key_here \
  -v $(pwd)/backtest_data:/app/backtest_data \
  stock-classifier
```

All outputs and the trained model are saved locally in `backtest_data/` (uploaded as a sample to repository as well).

---

## Model Description

* **Model:** RandomForest binary classifier
* **Task:** Predict future relative stock performance
* **Target variable:**
  * `1` = stock outperforms median future return
  * `0` = stock underperforms median future return
* **Features used:**
  * Volatility
  * Dividend yield
  * EPS growth

All features are computed from historical data available before the prediction window.

---

## Model Results

Evaluation performed on a balanced test set of 148 stocks.

```
ROC AUC: 0.636

              precision    recall  f1-score   support
0                 0.59      0.59      0.59        74
1                 0.59      0.58      0.59        74

accuracy                                 0.59       148
macro avg            0.59      0.59      0.59       148
weighted avg         0.59      0.59      0.59       148
```

### Interpretation

* ROC AUC ≈ **0.64** indicates **non-random predictive signal** in a noisy financial data set.
* Balanced precision and recall confirm:
  * No class bias
  * Stable generalization
* Performance is consistent with realistic financial ML benchmarks where values above 0.60 are considered strong.

## Model Usage Example (on Single Ticker)

The project includes a live inference function that demonstrates how the trained model can be used to generate a prediction for a **single stock** using fresh Yahoo Finance and FRED data.

---

### Example Output (AAPL)

```
Single ticker inference for AAPL
Volatility: 0.0171
DividendYield: 0.0507
EPSGrowth: 0.3070
Predicted OutperformFuture: 1
Predicted Probability: 0.9550
```

---

## How This Prediction Is Produced

For a single stock, the pipeline performs the following steps:

1. Downloads recent historical price data from Yahoo Finance
2. Computes volatility from daily returns
3. Computes dividend yield using total dividends and initial price
4. Computes EPS growth from quarterly reported earnings
5. Loads the trained RandomForest model
6. Produces:
   * A **binary classification**
   * A **probability of future outperformance**

---

## Interpretation of the Output

For the AAPL example:

| Output Field                     | Meaning                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| `Volatility = 0.0171`            | Historical daily risk of AAPL                                |
| `DividendYield = 0.0507`         | Income return from dividends                                 |
| `EPSGrowth = 0.3070`             | Annualized earnings growth                                   |
| `Predicted OutperformFuture = 1` | Model predicts AAPL will outperform the median future return |
| `Predicted Probability = 0.9550` | Model assigns a 95.5% probability to outperformance          |

* The model predicts that **AAPL belongs to the top 50% of future performers** in the selected stock set.
* The high probability indicates that, based on learned historical patterns, AAPL is combination of:

  * relatively low risk,
  * strong dividend yield,
  * and high earnings growth
    closely matches the profile of past outperforming stocks.


It is important to note that the model does not predict:

* Exact future price
* Exact future return
* Absolute profit or loss

Instead, it predicts **relative performance vs. peers (cross-sectional outperformance).**

---

## Key Engineering Challenges Solved

* API instability handling
* Delisted filtering
* Feature-target time isolation to avoid leakage
* NaN cascades prevention
* Cross-sectional feature alignment
* Reproducibility via Docker

---

## Fault Tolerance

* Delisted ticker filtering before ingestion
* Earnings availability check
* NaN handling for all numerical features
* Logging failure isolation
* Fallback for missing EPS growth
* Model training guards for insufficient data

---

## Project Structure

```text
.
├── stock_pipeline.py
├── sp500_10yr_range_tickers.csv
├── requirements.txt
├── Dockerfile
├── README.md
└── backtest_data/
```