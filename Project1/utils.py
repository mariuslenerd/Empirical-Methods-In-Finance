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
    plt.grid()

    return None


