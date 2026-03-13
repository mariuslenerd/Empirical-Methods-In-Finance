import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from itertools import permutations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import wrds
from statsmodels.tsa.stattools import adfuller
from utils import plot_wealth_positions_spread
from utils import plot_p_values
from utils import pnl_calculations
from statsmodels.tsa.stattools import coint as eg_coint


class Fetch_Data :
    def __init__(self,start_date,end_date,tickers): 
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers 
        self.data = None
    def download_data(self): 
        """
        Download data of interest using yahoo finance
        """
        self.data = yf.download(self.tickers, start = self.start_date, end = self.end_date)
        self.data = self.data['Close']
        return self.data

class Select_Pair: 
    def __init__(self,data): 
        self.data = data
    
    def permutations(self) : 
        """
        Extract all different possible permuations of pairs (ex : Booking - IHG, IHG-Booking,...)
        We need to test both ways because the fct cointegration first regresses Booking on IHG 
        (booking_t = alpha + beta IHG_t + epsilon_t) and then tests the residuals for stationarity. 

        The regression is the reason we cannot use one way to test cointegration, OLS is not symmetric (i.e regressing x on y is not the same as regressing y on x)
        this is due to the fact that beta_hat_y_knowing_x = Cov(x,y)/var(x) which is generally neq beta_hat_x_knowing_y = Cov(x,y)/Var(y)
        unless perfect linear correlation (i.e abs(rho) = 1). Also, a geometric explanation : ols minimizes vertical distance, not orthogonal distances
        """
        tickers = [i for i in self.data.columns]
        self.permutations = list(permutations(tickers,2))
        return self.permutations
    
    def are_cointegrated(self): 
        """
        Find the most cointegrated pair of assets. To do so, we employ the previously created pairs and 
        test each of them using the coint funciton from statsmodels.

        Note that we first restrict the length of the time series due to the fact that 1 asset has missing values until 2013. 
        To do so, we check what are the first and last value of each serie and force them to be the same (taking the maximum between 
        the 2 first values and the minimum between the 2 last values)
        """
        self.coint_results = {}
        for pair in self.permutations:
            ticker1 = pair[0]
            ticker2 = pair[1]

            price1 = self.data[ticker1].dropna()
            price2 = self.data[ticker2].dropna()

            start1 = price1.index[0]
            end1 = price1.index[-1]

            start2 = price2.index[0]
            end2 = price2.index[-1]

            if start1 != start2 : 
                    start = np.maximum(start1,start2)
            else : 
                    start = start1
            
            if end1 != end2 : 
                    end = np.minimum(end1,end2)
            else : 
                    end = end1

            price1 = price1[start:end]
            price2 = price2[start:end]

            score, pval, _ = coint(price1, price2)

            self.coint_results[(ticker1, ticker2)] = {'score': score, 'pvalue': pval}
        
        df_coint_results = pd.DataFrame(self.coint_results).transpose()

        self.most_coint_pair = df_coint_results['score'].idxmin()

        self.data_most_coint = self.data[[self.most_coint_pair[0], self.most_coint_pair[1]]].dropna()

        return self.most_coint_pair, self.data_most_coint
    
    def extract_ratios_cointegrated_pair(self,data_reg,tickers) : 
        """
        Extract alpha, beta and the residuals from a dataframe of the two assets we are regressing. 
        The first ticker corresponds to the y and the second ticker to the x

        """
        asset1 = data_reg[tickers[0]]
        asset2 = data_reg[tickers[1]]
        #reg : y = alpha + beta x + epsilon
        x = asset2
        x = sm.add_constant(x)
        y = asset1

        reg = sm.OLS(y,x,missing = 'drop').fit()
        self.alpha = float(reg.params['const'])
        self.beta = float(reg.params['IHG'])
        self.residuals = reg.resid

        return self.alpha, self.beta, self.residuals
    
    def normalize_residuals(self,resid): 
        """
         Standardize the residuals calculated in the function extract_ratios_cointegrated_pairs
        """
        mean = np.mean(resid)
        std = np.std(resid)
        self.norm_resid = (resid-mean)/std

        return self.norm_resid
    
    def test_stationarity(self) : 
        results = adfuller(self.norm_resid.dropna())

        p_val = results[1]
        crit_vals = results[4]

        return p_val, crit_vals
    def adf_test_results(self,p_val): 
        if p_val < 0.05 : 
             print("H0 is rejected : the residuals are stationary")
        else : 
             print("Fail to reject H0 : the residuals are non-stationary")
         
        
         

class Fetch_wrds: 
    """
    Creating a class in order to download bid and ask prices from Wrds database
    (access provided by EPFL; works as long as I'm a registered student)

    """
    def __init__(self,start_date, end_date,tickers,username): 
          self.start_date = start_date
          self.end_date = end_date
          self.tickers = tickers
          self.username = username
        
    def create_wrds_connection(self) : 
        self.db_connection = wrds.Connection(wrds_username = self.username)
    
    def fetch_bid_ask(self, ticker_aliases): 
        """
        Fetches daily bid and ask prices from CRSP via WRDS database.
        
        Note: crsp.dsf does not have a ticker column — it uses permno as primary key.
        We join crsp.dsf with crsp.dsenames to map tickers to permnos.

        Some companies have changed tickers over time (e.g. BKNG was PCLN before 2018).
        We resolve aliases via TICKER_ALIASES and relabel them back to the canonical ticker.

        If the data exists already in the Git folder, then it fetches it from there. If it doesn't, it loggs into
        WRDS database and downloads it. 
        """
        # Map tickers to all their historical aliases in CRSP
        # e.g. BKNG was PCLN before Feb 2018, so we need both to get full history
        #ticker_aliases = {'BKNG': ['BKNG', 'PCLN'],}

        # Expand tickers to include historical aliases
        self.ticker_aliases = ticker_aliases
        expanded = []
        alias_map = {}  # historical_ticker -> canonical_ticker
        for ticker in self.tickers:
            aliases = self.ticker_aliases.get(ticker, [ticker])
            for alias in aliases:
                expanded.append(alias)
                alias_map[alias] = ticker

        tickers_str = "'" + "','".join(expanded) + "'"

        query = f"""
            SELECT dsf.date, names.ticker, dsf.bid, dsf.ask
            FROM crsp.dsf AS dsf
            JOIN crsp.dsenames AS names
                ON dsf.permno = names.permno
            WHERE names.ticker IN ({tickers_str})
            AND dsf.date >= '{self.start_date}'
            AND dsf.date <= '{self.end_date}'
            AND dsf.date BETWEEN names.namedt AND COALESCE(names.nameendt, CURRENT_DATE)
            ORDER BY dsf.date, names.ticker
        """

        data = self.db_connection.raw_sql(query, date_cols=['date'])

        # Relabel historical tickers back to canonical ticker (e.g. PCLN -> BKNG)
        data['ticker'] = data['ticker'].map(lambda t: alias_map.get(t, t))

        # Pivot to MultiIndex columns: (ticker, bid/ask)
        data_bid_ask = data.pivot(index='date', columns='ticker', values=['bid', 'ask'])

        # Swap levels so ticker is first, then bid/ask
        data_bid_ask = data_bid_ask.swaplevel(axis=1).sort_index(axis=1)

        spread_BKNG = pd.Series(data_bid_ask['BKNG']['ask'] - data_bid_ask['BKNG']['bid']).rename('BKNG')
        spread_IHG = pd.Series(data_bid_ask['IHG']['ask'] - data_bid_ask['IHG']['bid']).rename('IHG')

        bid_ask_spread = pd.DataFrame([spread_BKNG,spread_IHG]).transpose()



        return bid_ask_spread
         

class Simple_Pair_Trading : 
    def __init__(self, data_raw,data_most_coint_pair,std_residuals,bid_ask_spread, alpha, beta, threshold) : 
        self.data_raw = data_raw
        self.alpha = alpha
        self.beta = beta
        self.spread = std_residuals
        self.bid_ask_spread = bid_ask_spread
        self.threshold = threshold
        self.data_most_coint_pair = data_most_coint_pair
        self.price_df = self.data_raw[self.data_most_coint_pair.columns].reindex(self.spread.index)
        self.return_df = self.price_df.pct_change()
    
    def simple_pair_trading(self) : 
        """
        This function is the most simple pair trading strategy that I will develop in this project.
        It is based on the full sample (look ahead bias for sure) and does not provide any backtesting of any sort
        The flaws are pretty straightforward, the whole analysis is based on alpha and beta measures of the whole sample
        which is not realistic : If we trade in 2013, we should have values estimated on the sample available until now, 
        not on the whole sample. It is useful however to familiarize myself with the different concepts and
        serves as a building block for what I will do subsequently. 

        The spread is the residual series defined as the difference between the observed value of asset A 
        and the equilibrium predicted value based on asset B. When this difference between what we should observe
        based on the long-term relationship and what we do observe diverges, this is where we enter a trade.  
         
        When this relation goes back to normal (spread is 0 again), we get out of the opened positions. 

        Based on recent litterature, an optimal value to enter a trade would be when the normalized spread 
        goes above |1.5|, then we enter the trade. 

        When the spread is >> 0, it means that asset A is overvalued (or asset B is undervalued), it should be 0 however it diverges from its 
        value and the spread is positive. This means that we should short asset A and long asset B because based on the statistical properties 
        of the spread, its value should revert and decrease in the coming times (or B increases). If the spread is <<0, the invert 
        takes place

        In this framework, we lever the mean-reverting property of the stationarity of the spread's time series. 

        Size of the position : P_a = alpha + beta P_b + epsilon --> epsilon = P_a - alpha - beta P_b
        Therefore, if spread > 1.5 : invest 1 in -P_a and beta in P_b, if spread < -1.5 : invest 1 in P_a and beta in -P_b

        So, summary of when we enter or exit : 
        Enter : 
            1) short_A, long_B (A overpriced, B underpriced) : If spread > 1.5 AND positions not opened already 
            2) long_A, short_B  If spread < -1.5 AND positions not opened already 
        Exit : 
            - If positions opened AND spread changes sign (crosses y = 0)
        """
        long_A = False
        short_A = False

        position_A = np.zeros(len(self.spread))
        position_B = np.zeros(len(self.spread))

        for i,(t,val) in enumerate(self.spread.items()) :
            if i > 0 : #carry forward previous positions 
                 position_A[i] = position_A[i-1]
                 position_B[i] = position_B[i-1]
            
            if not short_A and not long_A : #we are neither short A nor long A --> we have no position opened
                if val >= self.threshold : #if spread above threshold : open position short_A, long_B
                    position_A[i] = -1
                    position_B[i] = self.beta

                    short_A = True
    
                if val <= -self.threshold : #if spread below threshold : open position long_A, short_B
                    position_A[i] = 1
                    position_B[i] = -self.beta

                    long_A = True

            elif short_A and val<0: #we are short A --> we have opened an upward position --> close it if spread goes below 0 
                position_A[i] = 0
                position_B[i] = 0

                short_A = False

            elif long_A and val >= 0 : 
                position_A[i] = 0
                position_B[i] = 0

                long_A = False

        
        # Attach datetime index from std_residuals (positions come out as plain integer-indexed arrays)
        position_A = pd.Series(position_A, index=self.spread.index, name=self.data_most_coint_pair.columns[0])
        position_B = pd.Series(position_B, index=self.spread.index, name=self.data_most_coint_pair.columns[1])
        self.positions_df  = pd.concat([position_A, position_B], axis=1)

      
        cum_pnl,sharpe = pnl_calculations(self.positions_df,self.price_df,self.returns_df,self.spread, self.bid_ask_spread,self.threshold)
        return cum_pnl,sharpe
    

class Rolling_Pair_Trading : 
    def __init__(self, window, coint_window,data_raw, most_coint_pair_df, bid_ask_spread,threshold):
          self.window = window
          self.data_raw = data_raw
          self.most_coint_pair_df = most_coint_pair_df
          self.bid_ask_spread = bid_ask_spread
          self.price_df = self.data_raw[self.most_coint_pair_df.columns].reindex(self.bid_ask_spread.index)
          self.return_df = self.price_df.pct_change()
          self.threshold = threshold

    
    def extract_rolling_params(self) : 
        """
        Function responsible for extracting beta and estimating the spread based on a 252 days rolling window
        This prevents look ahead bias that was performed during the previous simple pair trading class
        The estimation window is 252 days.
        
        What happens is that during the first 252 days, we use this data to get a first
        estimate of alpha,beta and the spread by regressing BKNG price on IHG price. 
        We then normalize the spread by using the mean of the residuals as well as its std
        (based on the last 252 days only). We add them to a empty lists and go forward, etc
        until reaching the end of the overall period. 
        - Args : self
        - Returns : None
        """

        self.tickers_pair = list(self.most_coint_pair_df.columns) #fetch tickers from the most cointegrated pair
        self.ticker_A, self.ticker_B = self.tickers_pair[0], self.tickers_pair[1] 
        n = len(self.most_coint_pair_df) #nb of rows 

        # Now we create empty series for storing rolling spread and betas
        # We do not trade during the first window : it serves for estimating the params
        rolling_spread = pd.Series(np.nan, index=self.most_coint_pair_df.index)
        rolling_beta   = pd.Series(np.nan, index=self.most_coint_pair_df.index)

        for t in range(self.window, n): #252 --> 3'774
        # Estimation window : We fetch the data for the current period --> [0:252], [1:253],..., [3522:3774]
            estimation_data = self.most_coint_pair_df.iloc[t - self.window : t]

            selectpair_roll = Select_Pair(estimation_data)
            self.alpha, beta, resid = selectpair_roll.extract_ratios_cointegrated_pair(estimation_data, self.tickers_pair)

            # Now we can estimate the spread : 
            # BKNGG_t = alpha + beta IHG_t + eps_t where alpha and beta 
            # Are based on the last 252 days until yesterday but we use todays price 
            # Based on the predicted equilibrium price (which is based on last 252 days), we
            # trade if the actual price deviates from its equilibrium
            spread_t = self.most_coint_pair_df[self.ticker_A].iloc[t] - self.alpha - beta * self.most_coint_pair_df[self.ticker_B].iloc[t]

            # Now we normalize today's spread using the last 252 days mean resid and std
            rolling_spread.iloc[t] = (spread_t - resid.mean()) / resid.std()
            rolling_beta.iloc[t]   = beta

        # Drop the NaN warm-up period
        self.rolling_spread_clean = rolling_spread.dropna()
        self.rolling_beta_clean   = rolling_beta.dropna()

        return None

    def simple_rolling_pair_trading(self) : 
        """
         Fct responsible for the pair trading strategy where parameters are estimated based on a 1y window
         Prevents look ahead bias (when using future data for today's decision)

            - Args : self
            - Returns : None
        """
         # Rolling trading loop
         # Same logic as simple_pair_trading but with time-varying beta for position sizing and time-varying spread 
         # for entry/exit signals

        long_A = False
        short_A = False

        pos_A = np.zeros(len(self.rolling_spread_clean))
        pos_B = np.zeros(len(self.rolling_spread_clean))

        for i, (t, val) in enumerate(self.rolling_spread_clean.items()):
            b = self.rolling_beta_clean[t]   # beta estimated on past window : used for position size

            if i > 0:   # carry forward
                pos_A[i] = pos_A[i-1]
                pos_B[i] = pos_B[i-1]

            if not short_A and not long_A:
                if val >= self.threshold:            # spread too high : short A, long B
                    pos_A[i] = -1
                    pos_B[i] =  b
                    short_A = True
                elif val <= -self.threshold:         # spread too low : long A, short B
                    pos_A[i] =  1
                    pos_B[i] = -b
                    long_A = True
            elif short_A and val <= 0:           # spread reverted : close
                pos_A[i] = 0
                pos_B[i] = 0
                short_A = False
            elif long_A and val >= 0:           # spread reverted : close
                pos_A[i] = 0;  pos_B[i] = 0
                long_A = False

        # Build positions DataFrame with datetime index 
        rolling_pos_A = pd.Series(pos_A, index=self.rolling_spread_clean.index, name=self.ticker_A)
        rolling_pos_B = pd.Series(pos_B, index=self.rolling_spread_clean.index, name=self.ticker_B)
        self.rolling_positions_df = pd.concat([rolling_pos_A, rolling_pos_B], axis=1)

        # PnL using the same method as before
        self.rolling_price_df   = self.price_df[self.tickers_pair].reindex(self.rolling_spread_clean.index) #re-index the original data to match
        self.rolling_returns_df = self.rolling_price_df.pct_change()
        self.rolling_bid_ask_spread_df  = self.bid_ask_spread.reindex(self.rolling_spread_clean.index)

        cum_pnl,sharpe = pnl_calculations(self.rolling_positions_df,self.rolling_price_df,self.rolling_returns_df,self.rolling_spread_clean, self.rolling_bid_ask_spread_df,self.threshold)

        return cum_pnl,sharpe


                

class Rolling_Pair_Trading_coint_filter : 
    """
        Now, in order to feed even stronger signals to the trading strategy, I will apply another layer 
        of filtering. There is a new rolling window which is called cointegration window. This window
        checks the cointegration (based on last 504 days of data) using a Engle Granger test which 
        tests whether the process has a unit root (non stationary). If the process has a unit root, 
        then the 2 assets are not cointegrated based on the last 504 observations and therefore,
        the trading process is not allowed. if there are positions opened, they are closed and we can repoen 
        when cointegration is "accepted" again. 
    """
    def __init__(self,significance_level,window,coint_window,data_raw, most_coint_pair_df, bid_ask_spread,threshold):
        self.significance_level = significance_level
        self.window = window # 1-year window for alpha/beta/spread (in order to adapt faster to regime changes)
        self.coint_window = coint_window # 2-year window for cointegration test (longer sample = more test power = fewer false "not cointegrated" rejections)
        self.data_raw = data_raw
        self.most_coint_pair_df = most_coint_pair_df
        self.bid_ask_spread = bid_ask_spread
        self.price_df = self.data_raw[self.most_coint_pair_df.columns].reindex(self.bid_ask_spread.index)
        self.return_df = self.price_df.pct_change()
        self.threshold = threshold  

    def extract_cointegration_filter_params(self) : 
        """
        This function is responsible for extracting the parameters (spread,beta,cointegration pvalue)
        on each rolling window. 

        The rolling window for extracting the alpha, beta and the spread (residuals of the regression) 
        is based on the last 252 observations. The cointegration window for extracting the p-values 
        is based on 504 observations for stronger estimations .
        """      
        self.tickers_pair = list(self.most_coint_pair_df.columns) #fetch tickers from the most cointegrated pair
        self.ticker_A, self.ticker_B = self.tickers_pair[0], self.tickers_pair[1] 
        n = len(self.most_coint_pair_df) #nb of rows 

        # Now we create empty series for storing rolling spread, betas and cointegration p-values
        self.rolling_spread = pd.Series(np.nan, index=self.most_coint_pair_df.index)
        self.rolling_beta   = pd.Series(np.nan, index=self.most_coint_pair_df.index)
        rolling_coint_pval = pd.Series(np.nan, index = self.most_coint_pair_df.index)


        # Start at window_coint (the larger window) so both sub-windows always have enough data
        # We do not trade during the first window : it serves for estimating the params
        for t in range(self.coint_window, n):
            coint_data = self.most_coint_pair_df.iloc[t - self.coint_window : t]  # 504-day window for EG test
            beta_data  = self.most_coint_pair_df.iloc[t - self.window : t]  # 252-day window for OLS

            # 1) Engle-Granger cointegration test on the 504-day window
            _, pval, _ = eg_coint(coint_data[self.ticker_A], coint_data[self.ticker_B])
            rolling_coint_pval.iloc[t] = pval

            # 2) estimate alpha, beta, spread on the 252-day window
            selectpair_cf = Select_Pair(beta_data)
            alpha_cf, beta_cf, resid_cf = selectpair_cf.extract_ratios_cointegrated_pair(beta_data, self.tickers_pair)

            # Out-of-sample spread for day t (yesterday's model applied to today's price)
            spread_t = self.most_coint_pair_df[self.ticker_A].iloc[t] - alpha_cf - beta_cf * self.most_coint_pair_df[self.ticker_B].iloc[t]
            self.rolling_spread.iloc[t] = (spread_t - resid_cf.mean()) / resid_cf.std()
            self.rolling_beta.iloc[t]   = beta_cf

        # summary 
        self.rolling_coint_pval_clean = rolling_coint_pval.iloc[self.coint_window:]
        n_total = len(self.rolling_coint_pval_clean)
        n_coint = int((self.rolling_coint_pval_clean < self.significance_level).sum())
        pct     = 100 * n_coint / n_total
        print(f"Cointegrated windows : {n_coint} / {n_total} ({pct:.1f}%) → trading allowed")
        print(f"Non-cointegrated     : {n_total - n_coint} ({100 - pct:.1f}%) → forced flat")
        
        plot_p_values(rolling_coint_pval,self.significance_level,self.coint_window)

        return None

    def cointegration_filter_pair_trading(self) : 
        # 
        long_A  = False
        short_A = False

        post_warmup_index = self.most_coint_pair_df.index[self.coint_window:] #all rows after the first window which is unsable for trading 
        #as we need a first estimate cointegration
        n_trading = len(post_warmup_index)

        pos_A = np.zeros(n_trading)
        pos_B = np.zeros(n_trading)

        for i, date in enumerate(post_warmup_index):

            # Carry forward previous day's position
            if i > 0:
                pos_A[i] = pos_A[i-1]
                pos_B[i] = pos_B[i-1]

            # Cointegration condition : if p-value >= threshold : not cointegrated --> close positions and positions stay flat
            if self.rolling_coint_pval_clean.loc[date] >= self.significance_level:
                if long_A or short_A:           # forced exit when relationship breaks down
                    pos_A[i] = 0;  pos_B[i] = 0
                    long_A = False;   short_A = False
                continue   # no new signal : go to next timestep

            val = self.rolling_spread.loc[date]
            b   = self.rolling_beta.loc[date]

            # Standard spread trading conditions (same as previously)
            if not short_A and not long_A:
                if val >= self.threshold:            # spread too high --> short A, long B
                    pos_A[i] = -1 
                    pos_B[i] =  b  
                    short_A = True
                elif val <= -self.threshold:         # spread too low  --> long A, short B
                    pos_A[i] =  1;  pos_B[i] = -b;  long_A  = True
            elif short_A and val <= 0:           # mean-reverted --> close
                    pos_A[i] = 0
                    pos_B[i] = 0
                    short_A = False
            elif long_A  and val >= 0:          # mean-reverted → close
                    pos_A[i] = 0
                    pos_B[i] = 0
                    long_A  = False

        # Positions df
        pos_A = pd.Series(pos_A, index=post_warmup_index, name=self.ticker_A)
        pos_B = pd.Series(pos_B, index=post_warmup_index, name=self.ticker_B)
        self.positions_df = pd.concat([pos_A, pos_B], axis=1)

        # PnL 

        self.price_df   = self.data_raw[self.tickers_pair].reindex(post_warmup_index)
        self.returns_df = self.cf_price_df.pct_change()
        self.spread_df  = self.bid_ask_spread.reindex(post_warmup_index)

        cum_pnl,sharpe = pnl_calculations(self.positions_df,self.price_df,self.returns_df,self.spread, self.bid_ask_spread,self.threshold)

        return cum_pnl,sharpe





