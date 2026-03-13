import argparse
from itertools import permutations

# --- Imports ---
from data_and_trading_utils import Fetch_Data
from data_and_trading_utils import Select_Pair
from data_and_trading_utils import Fetch_wrds
from data_and_trading_utils import Simple_Pair_Trading
from data_and_trading_utils import Rolling_Pair_Trading
from data_and_trading_utils import Rolling_Pair_Trading_coint_filter
import numpy as np
from utils import plot_n_series
import pandas as pd
	

# --------- Parameters ----------
start_date = '2010-01-01'
end_date = '2025-01-01'
tickers = ['IHG', 'HLT', 'MAR', 'BKNG', 'H']
wrds_username = 'mariuspecaut'


def run_full_pipeline(
	start_date,
	end_date,
	tickers,
	wrds_username,
	threshold=1.5,
	coint_pvalue_threshold=0.05,
	window=252,
	coint_window=504,
	use_wrds=True,):

	print("Step 1/6: Downloading data...")
	fetcher = Fetch_Data(start_date, end_date, tickers)
	data_raw = fetcher.download_data()
	plot_n_series(data_raw, "Stock Prices (Log Scale)", "log", "Date", "Price (log scale)")

	print("Step 2/6: Selecting cointegrated pair...")
	data = np.log(data_raw)
	pairselect = Select_Pair(data)
	pairselect.permutations()
	most_coint_pair, data_most_coint_pair = pairselect.are_cointegrated()
	print(f"Most cointegrated pair: {most_coint_pair[0]} - {most_coint_pair[1]}")

	tickers_pair = [data_most_coint_pair.columns[0], data_most_coint_pair.columns[1]]
	alpha, beta, residuals = pairselect.extract_ratios_cointegrated_pair(data_most_coint_pair, tickers_pair)
	std_residuals = pairselect.normalize_residuals(residuals)

	print("Step 3/6: Building bid-ask spread input...")
	tickers_wrds = [tickers[0],tickers[3]]
	fetch_wrds = Fetch_wrds(start_date, '2025-02-01', tickers_wrds, wrds_username)
	fetch_wrds.create_wrds_connection()
	ticker_aliases = {'BKNG': ['BKNG', 'PCLN'],}
	bid_ask_spread = fetch_wrds.fetch_bid_ask(ticker_aliases)

	print("Step 4/6: Running simple pair trading...")
	print("First testing whether the residusals of the 2 most cointegrated assets are stationary :")
	p_val,crit_vals = pairselect.test_stationarity()
	pairselect.adf_test_results(p_val)
	print("Now running the first simple pair trading strategy : ")
	simplepairtrading = Simple_Pair_Trading(data_raw,data_most_coint_pair,std_residuals,bid_ask_spread,alpha,beta,threshold)
	cumulative_pnl_simple, sharpe_ratio_simple = simplepairtrading.simple_pair_trading()

	print("Step 5/6: Running rolling window pair trading for preventing look ahead bias...")
	rollingpairtrading = Rolling_Pair_Trading(window, coint_window, data_raw, data_most_coint_pair, bid_ask_spread, threshold)
	rollingpairtrading.extract_rolling_params()
	cumulative_pnl_rolling, sharpe_ratio_rolling = rollingpairtrading.simple_rolling_pair_trading()

	print("Step 6/6: Running rolling strategy with cointegration filter ; Prevents from trading when last 252 are not cointegrated ...")
	rollingcointfilter_pairtrading = Rolling_Pair_Trading_coint_filter(coint_pvalue_threshold,window,coint_window,data_raw,data_most_coint_pair,bid_ask_spread,threshold)
	rollingcointfilter_pairtrading.extract_cointegration_filter_params()
	cumulative_pnl_rolling_cointfilter, sharpe_ratio_rolling_cointfilter = (rollingcointfilter_pairtrading.cointegration_filter_pair_trading())



	results = pd.DataFrame(
		{
			"strategy": ["simple", "rolling", "rolling_coint_filter"],
			"sharpe": [
				sharpe_ratio_simple,
				sharpe_ratio_rolling,
				sharpe_ratio_rolling_cointfilter,
			],
			"final_cum_pnl": [
				float(cumulative_pnl_simple.iloc[-1]),
				float(cumulative_pnl_rolling.iloc[-1]),
				float(cumulative_pnl_rolling_cointfilter.iloc[-1])
			],
		}
	)
	print("\n=== Strategy summary ===")
	print(results.to_string(index=False))
	return results


def _parse_args():
	parser = argparse.ArgumentParser(description="Run full pairs-trading project pipeline from terminal.")
	parser.add_argument("--start-date", default="2010-01-01")
	parser.add_argument("--end-date", default="2025-01-01")
	parser.add_argument("--tickers", default="IHG,HLT,MAR,BKNG,H")
	parser.add_argument("--threshold", type=float, default=1.5)
	parser.add_argument("--window", type=int, default=252)
	parser.add_argument("--coint-window", type=int, default=504)
	parser.add_argument("--coint-pvalue-threshold", type=float, default=0.05)
	parser.add_argument("--wrds-username", default=None)
	parser.add_argument("--no-wrds", action="store_true")
	return parser.parse_args()


def main():
	args = _parse_args()
	tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
	run_full_pipeline(
		start_date=args.start_date,
		end_date=args.end_date,
		tickers=tickers,
		wrds_username=args.wrds_username,
		threshold=args.threshold,
		coint_pvalue_threshold=args.coint_pvalue_threshold,
		window=args.window,
		coint_window=args.coint_window,
		use_wrds=not args.no_wrds,
	)


if __name__ == "__main__":
	main()

