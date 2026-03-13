# Pairs Trading Project

This repository contains a personnal project about pairs trading. The objective is to build and compare a few pairs-trading variants on hotel stocks, from raw data to strategy evaluation.

## What the project does

The script runs a complete pipeline:

1. Download daily prices from Yahoo Finance.
2. Test all ticker pairs and select the most cointegrated one.
3. Estimate spread/hedge ratio with OLS.
4. Run three strategies (simple, rolling, rolling + cointegration filter).
5. Compute cumulative PnL and annualized Sharpe ratio.
6. Print a summary table to compare results.

## Project structure

```
Empirical-Methods-In-Finance/
├── README.md
└── Project1/
        ├── main.py
        ├── data_and_trading_utils.py
        ├── utils.py
        └── visualization.ipynb
```

- `main.py`: terminal entry point (`argparse`) and pipeline orchestration.
- `data_and_trading_utils.py`: classes for data, pair selection, and strategy logic.
- `utils.py`: plotting and shared PnL helpers.
- `visualization.ipynb`: notebook version used for step-by-step visual checks.

## Requirements

Install dependencies from the project root:

```bash
pip install numpy pandas matplotlib yfinance statsmodels wrds
```

## Run from terminal

From the repository root:

```bash
cd Project1
python main.py --wrds-username YOUR_WRDS_USERNAME
```

Example with custom parameters:

```bash
python main.py \
    --start-date 2012-01-01 \
    --end-date 2025-01-01 \
    --tickers IHG,HLT,MAR,BKNG,H \
    --threshold 1.5 \
    --window 252 \
    --coint-window 504 \
    --coint-pvalue-threshold 0.05 \
    --wrds-username YOUR_WRDS_USERNAME
```

To list all available options:

```bash
python main.py --help
```

## Notes

- WRDS access is used to retrieve bid/ask data for transaction-cost estimation.
- Booking ticker aliases are handled in the WRDS query (`BKNG` and historical `PCLN`).
