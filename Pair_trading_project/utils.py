import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def plot_n_series(data,title,yscale,xlabel,ylabel) : 
    """
    function to plot n series simultaneously. I use a function such that all plots in the notebook will have the same format/style
    """
    plt.figure(figsize=(12,6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for col in data.columns : 
        plt.plot(data.index, data[col], label = col)
    plt.legend(title = 'Ticker', loc = 'best')
    plt.yscale('log')
    plt.grid()

    return None

def plot_wealth_positions_spread(data,residuals, threshold,positions,cumulative_pnl) : 
    ticker_A = data.columns[0]
    ticker_B = data.columns[1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Normalised spread + thresholds + shaded positions 
    axes[0].plot(residuals, color='gray', linewidth=1, label='z-score (spread)')
    axes[0].axhline( threshold, color='red',   linewidth=1.2, linestyle='--', label=f'Entry threshold (+{threshold})')
    axes[0].axhline(-threshold, color='green', linewidth=1.2, linestyle='--', label=f'Entry threshold (-{threshold})')
    axes[0].axhline(0,          color='black', linewidth=0.8)
    axes[0].set_title('Normalised Spread (z-score) & Positions', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('z-score')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Position sizes over time 
    axes[1].step(positions.index, positions[ticker_A], where='post', color='steelblue',
             linewidth=1.5, label=f'Position {ticker_A} (+1=long, -1=short)')
    axes[1].step(positions.index, positions[ticker_B], where='post', color='darkorange',
             linewidth=1.5, label=f'Position {ticker_B} (+β=long, -β=short)', linestyle='--')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Positions Over Time', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Position size')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(alpha=0.3)

    # Cumulative PnL 
    axes[2].plot(cumulative_pnl.index, cumulative_pnl, color='purple', linewidth=1.5, label='Cumulative net PnL')
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].fill_between(cumulative_pnl.index, cumulative_pnl, 0,
                     where=(cumulative_pnl >= 0), alpha=0.15, color='green', label='Profit')
    axes[2].fill_between(cumulative_pnl.index, cumulative_pnl, 0,
                     where=(cumulative_pnl < 0),  alpha=0.15, color='red',   label='Loss')
    axes[2].set_title('Cumulative PnL (fraction of notional)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Cumulative PnL')
    axes[2].set_xlabel('Date')
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_p_values(pvalues,significance_level, window_coint) : 
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(pvalues.index, pvalues.values,
        lw=0.8, color='steelblue', label='EG p-value')
    ax.axhline(significance_level, color='red', lw=1.2, linestyle='--',
           label=f'Threshold ({significance_level})')
    ax.fill_between(pvalues.index, 0, 1,
                where=(pvalues >= significance_level),
                alpha=0.15, color='red', label='Not cointegrated (flat)')
    ax.set_ylim(0, 1)
    ax.set_ylabel('p-value')
    ax.set_title(f'Rolling Engle-Granger Cointegration p-value ({window_coint}-day window)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 

def pnl_calculations(positions_df,price_df,returns_df,spread_df,bid_ask_spread_df,threshold):
        """
        Function that calculates the PnL of a strategy based on the positions taken on asset A and on asset B 
        In order to make it as realistic as possible, I take into account the bid-ask spread (buy at bid, sell at ask)
        as well as fixed transaction costs as a function of the price 
        Args : 
            - positions_df (pd.DataFrame) : df of positions taken through time in the 2 assets
            - price_df (pd.DataFrame) : df of prices through time for the 2 assets
            - returns_df (pd.DataFrame) : df of returns through time for the 2 assets
            - spread_df (pd.DataFrame) : df of the spread extracted from OLS regression and the log-price of the 2 assets
            - bid_ask_spread_df (pd.DataFrame) : df of the bid and ask for the 2 assets through time
            - threshold (float) : scalar that designates the threshold at which we open positions
            
        Returns : 
            - cum_pnl (pd.DataFrame) : df of the evolution of the cumulative PnL 
            - Sharpe ratio (float)
        """
        lagged_positions = positions_df.shift(1).fillna(0)
        position_changes = positions_df.diff().fillna(0)
        raw_pnl = lagged_positions.values*returns_df.values

        pnl = raw_pnl.sum(axis = 1)

        transaction_cost_pct = (bid_ask_spread_df/2)/price_df
        transactions_costs = (transaction_cost_pct*np.abs(position_changes.values)).sum(axis=1)


        net_pnl = pnl-transactions_costs
        cum_pnl = net_pnl.cumsum()

        #Calculating sharpe ratio :
        #Assumption : Rf = 5% --> daily rf = 0.05/252
        rf_daily = 0.05/252
        excess_pnl = net_pnl - rf_daily

        sharpe_ratio = (excess_pnl.mean()/excess_pnl.std())*np.sqrt(252)
        

        # Plot using previously coded fct 
        plot_wealth_positions_spread(
            price_df.reindex(spread_df.index),
            spread_df, threshold,positions_df, cum_pnl)
        print(f"Sharpe ratio (rolling, annualised): {sharpe_ratio:.4f}")
        return cum_pnl,sharpe_ratio


        


