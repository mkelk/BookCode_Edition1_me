import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import arviz as az

from xarray import DataArray, Dataset

def get_distict_color(i, map_name='tab10'):
    """
    Get a distinct color for a given index
    Args:
        i (int): index
        map_name (str): colormap name
        
    Returns:
        str: color
    """
    cmap = plt.get_cmap(map_name, 10)         # Get the colormap and 10 colors from it
    rgba_color = cmap(i)                     # Get the ith color from the colormap
    hex_color = mcolors.rgb2hex(rgba_color)  # Convert the RGBA color to hexadecimal format
    return hex_color           

def line_plot(dataframe: pd.DataFrame, features: list, title: str, normalize: bool = True) -> None:
    """
    Plots multiple time series
    Args:
        dataframe (pd.DataFrame): data
        features (list): features to plot
        title (str): title for chart
    """

    colors = {0: '#070620', 1: '#dd4fe4'}
    
    plt.rcParams["figure.figsize"] = (20,3)
    for i in range(len(features)):
        if normalize:
            dataframe[[features[i]]] = MinMaxScaler().fit_transform(dataframe[[features[i]]])
        sns.lineplot(data=dataframe, x='ds', y=features[i], label=features[i], color=get_distict_color(i))
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Values')
    plt.show()




class MyModel():
    def __init__(self, data, targetvar: str, timevar: str, features: list = None):
        self.data = data
        self.targetvar = targetvar
        self.timevar = timevar
        self.features = features

    def line_plot_general(self, features: list, title: str, normalize: bool = True) -> None:
        line_plot(self.data.copy(), features, title, normalize)

    def plot_channel_spends(self, normalize: bool = False) -> None:
        """Plot all channel spends on same chart"""
        self.line_plot_general(self.features, 'Channel Spends', normalize)

    def plot_sales(self, normalize: bool = False) -> None:
        """Plot sales in original data"""
        self.line_plot_general([self.targetvar], 'Sales', normalize)

    def plot_channel_spends_vs_sales(self) -> None:
        """Plot channel spends vs sales for each channel and shown normalized"""
        for feature in self.features:
            self.line_plot_general([feature] + [self.targetvar], f'{feature} spend vs Sales')

    def plot_correlation_matrix(self) -> None:
        """Plot correlation matrix between channel spends and sales"""
        corr_matrix = self.data[self.features + [self.targetvar]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='Blues')
        plt.show()
