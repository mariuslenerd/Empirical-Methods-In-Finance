# 📈 Pairs Trading — Empirical Methods in Finance
## Project 1 | Hotel Industry Stocks

---

## 📋 Overview

This project implements a **pairs trading pipeline** on hotel industry stocks. It identifies the most cointegrated pair of assets using the Engle–Granger cointegration test, extracts the regression residuals (spread), and normalizes them for use in a trading strategy.

**Stocks covered:** Booking Holdings (`BKNG`), Hyatt (`H`), Hilton (`HLT`), Marriott (`MAR`), InterContinental Hotels Group (`IHG`)  
**Data source:** Yahoo Finance via `yfinance`  
**Sample period:** 2010–2025

---

## 📁 Project Structure

```
Project1/
├── main.py                # Core classes: Fetch_Data and Select_Pair
├── utils.py               # Plotting utility: plot_n_series
├── visualization.ipynb    # Main notebook — run the full pipeline here
└── README.md
```

---

## ⚙️ Dependencies

Install all required packages via pip:

```bash
pip install numpy pandas matplotlib yfinance statsmodels wrds
```

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `yfinance` | Download stock price data |
| `statsmodels` | Cointegration test (Engle–Granger) & OLS regression |
| `wrds` | Access to WRDS financial database (optional) |

---

## 🚀 How to Run

Open and run **`visualization.ipynb`** from top to bottom. The notebook is self-contained and calls the classes defined in `main.py` and the plotting function from `utils.py`.

### Step-by-step

**1. Set parameters** (cell 6)
```python
start_date = '2010-01-01'
end_date   = '2025-02-01'
tickers    = ['IHG', 'HLT', 'MAR', 'BKNG', 'H']
```

**2. Download data** — uses `Fetch_Data` from `main.py`
```python
fetcher = Fetch_Data(start_date, end_date, tickers)
data    = fetcher.download_data()   # returns log(Close prices)
```

**3. Find the most cointegrated pair** — uses `Select_Pair` from `main.py`
```python
pairselect = Select_Pair(data)
permut     = pairselect.permutations()   # all ordered pairs
most_coint_pair, data_most_coint_pair = pairselect.are_cointegrated()
```

**4. Extract regression coefficients & residuals**
```python
alpha, beta, residuals = pairselect.extract_ratios_cointegrated_pair(
    data_most_coint_pair, most_coint_pair
)
```

**5. Normalize the spread**
```python
norm_resid = pairselect.normalize_residuals(residuals)
```

---

## 🧩 Module Reference

### `main.py`

#### `Fetch_Data(start_date, end_date, tickers)`
Downloads and log-transforms closing prices from Yahoo Finance.

| Method | Returns | Description |
|--------|---------|-------------|
| `download_data()` | `pd.DataFrame` | Log-transformed Close prices for all tickers |

---

#### `Select_Pair(data)`
Runs the cointegration analysis on all ordered pairs.

| Method | Returns | Description |
|--------|---------|-------------|
| `permutations()` | `list of tuples` | All ordered pairs of tickers (both directions) |
| `are_cointegrated()` | `(tuple, DataFrame)` | Most cointegrated pair and its price data |
| `extract_ratios_cointegrated_pair(data, tickers)` | `(α, β, residuals)` | OLS regression coefficients and residuals |
| `normalize_residuals(residuals)` | `pd.Series` | Z-score normalized spread |

> **Why both directions?** The Engle–Granger test runs an OLS regression in step 1. OLS is not symmetric — regressing $y$ on $x$ differs from regressing $x$ on $y$ because $\hat{\beta}_{y|x} = \frac{\text{Cov}(x,y)}{\text{Var}(x)} \neq \frac{\text{Cov}(x,y)}{\text{Var}(y)} = \hat{\beta}_{x|y}$. Testing both directions ensures we do not miss a cointegrated pair.

---

### `utils.py`

#### `plot_n_series(data, title, yscale, xlabel, ylabel)`
Plots multiple time series on the same figure with a legend.

```python
plot_n_series(data, 'Stock Prices (Log Scale)', 'log', 'Date', 'Price (log scale)')
```

---

## 📐 Methodology

1. **Log prices** — all prices are log-transformed before analysis. Log prices preserve non-stationarity (needed for cointegration) while making the spread economically meaningful (percentage-based).

2. **NaN handling** — one stock (HLT) starts trading later than the others. Each pair is aligned to the common observation window before testing.

3. **Cointegration test** — the Engle–Granger two-step test (`statsmodels.tsa.stattools.coint`):
   - Step 1: OLS regression $y_t = \alpha + \beta x_t + \epsilon_t$
   - Step 2: ADF test on residuals $\epsilon_t$. If residuals are stationary → pair is cointegrated.

4. **Pair selection** — the pair with the most negative ADF score (strongest stationarity evidence) is selected.

5. **Spread normalization** — residuals are z-score normalized: $z_t = \frac{\epsilon_t - \mu}{\sigma}$, used to generate trading signals.

---
