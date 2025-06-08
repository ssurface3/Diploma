# SHORT TERM
import subprocess
import sys
sys.path.append("..")  # Adjust if notebooks are deeper

# from dataloader import Data 
"""
Decided to skip data importation and 


"""
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
install_package("catboost")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd
import numpy as np
from typing import Literal
from YieldCurve import shortterm, dataloader
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
class pictures(): 
    def __init__(self ,csv_path):
        # self.dl = dataloader.Dataloader()
        self.short_term_rate = shortterm.short_term_rate(csv_path)
        self.dl_  = dataloader.DataLoader(csv_path)
        """
        Here i initilize models so i don't have to manage if statements if there is a new one 
        """
        self._dispatch_map_short_term = {
            'HullWhite_timedependant_nojump': self.short_term_rate.HullWhite_timedependant_nojump,
            'HullWhite_timedependant_jump': self.short_term_rate.HullWhite_timedependant_jump,
            'Vasichek': self.short_term_rate.Vasichek,
            'HullWhite_timedependant_nojump_constant_vol' : self.short_term_rate.HullWhite_timedependant_nojump_constant_vol
        }        
    def image_to_pdf(sefl,figure:object, 
                     output_pdf_path:str = os.path.join(os.path.join(os.path.expanduser("~"), "Desktop")) # just a desktop
                     ) -> None:
        """
        Saves a figure directly to a PDF file.
        --------------------------------------
        Can save an existing object to a specified path
            # matplotlib graph object 
        
        Parameters:
        - figure (matplotlib.figure.Figure): The figure to save.
        - output_pdf_path (str): Path for the output PDF file. or will just download it do a desktop
        
        Returns:
        - str: Confirmation message or error.
        """
        try:
            with PdfPages(output_pdf_path) as pdf:
                pdf.savefig(figure, bbox_inches='tight')
            return f"Successfully saved figure to '{output_pdf_path}'."
        except Exception as e:
            return f"Error: {str(e)}"
    def __nelson_siegel_svensson(self,tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
        term1 = (1 - np.exp(-lambda1 * tau)) / (lambda1 * tau)
        term2 = term1 - np.exp(-lambda1 * tau)
        term3 = (1 - np.exp(-lambda2 * tau)) / (lambda2 * tau) - np.exp(-lambda2 * tau)
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
    # def draw_nss(self):
    #     self.params_nss , _ = curve_fit(self.__nelson_siegel_svensson())

    #     # # Example data: maturities (in years) and observed yields
    #     # tau = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    #     # yields = np.array([0.01, 0.012, 0.013, 0.015, 0.016, 0.018, 0.019, 0.02, 0.021, 0.022])

    #     # # Initial parameter guess: [beta0, beta1, beta2, beta3, lambda1, lambda2]
    #     # p0 = [0.02, -0.01, 0.01, 0.01, 1.0, 3.0]

    #     # Fit the model
    #     params, _ = curve_fit(nelson_siegel_svensson, tau, yields, p0=p0, maxfev=10000)

    #     # Generate fitted curve
    #     tau_fit = np.linspace(0.01, 30, 300)
    #     y_fit = nelson_siegel_svensson(tau_fit, *params)

    #     # Plot
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(tau, yields, 'o', label='Observed Yields')
    #     plt.plot(tau_fit, y_fit, label='Nelson-Siegel-Svensson Fit')
    #     plt.xlabel('Maturity (years)')
    #     plt.ylabel('Yield')
    #     plt.title('Nelson-Siegel-Svensson Yield Curve Fit')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    #     print("Fitted parameters:", params)
    def draw(self, true_rate:pd.Series , 
             simulated_rates:pd.Series , 
             start_date:str , 
             dates:pd.Series, 
             name_of_model:str, 
             upper_quantile:int = 5, 
             downer_quantile:int = 95,
             plot_q = False
             ):
        """
        Plot the comparison between real interest rate data and simulated rate paths from a model.

        Parameters
        ----------
        true_rate : pd.Series
            The observed real interest rates indexed by date.
        simulated_rates : pd.Series or pd.DataFrame
            Simulated interest rate paths. If DataFrame, each column is a simulation.
            Indexed by date or time steps corresponding to `dates`.
        start_date : str
            The start date of the simulation/observation period (format: 'YYYY-MM-DD').
        dates : pd.Series or array-like
            The dates or time points corresponding to the rates for plotting on the x-axis.
        name_of_model : str
            Name of the model used for simulation (e.g., 'Vasichek', 'HW', etc.). Used for plot title and labels.
        upper_quantile : int, optional, default=95
            The upper percentile for quantile shading (e.g., 95 for 95th percentile).
        downer_quantile : int, optional, default=5
            The lower percentile for quantile shading (e.g., 5 for 5th percentile).
        plot_q : bool, optional, default=False
            Whether to plot the quantile range (shaded area) between `downer_quantile` and `upper_quantile`.
        ax : matplotlib.axes.Axes, optional, default=None
            Matplotlib Axes object to plot on. If None, a new figure and axes are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object containing the plot.

        Notes
        -----
        - The mean of the simulated rates is plotted as the central simulation tendency.
        - If `plot_q` is True, a shaded region between the specified quantiles is shown to visualize uncertainty.
        - The x-axis is labeled with the provided `dates` series.
        - The plot title includes the model name for clarity.
        """
        # quantile calculation
        shape = simulated_rates.shape
        if len(shape) > 1 and shape[1] > 1:
            mean_ = simulated_rates.mean(axis= 1)
        else:
            mean_ = simulated_rates
        

        # plotting

        
        fig, ax = plt.subplots(figsize=(12, 6))
        min_len = min(len(mean_), len(true_rate))
        true_rate_aligned, simulated_rates_aligned = true_rate.align(mean_, join='inner', axis=0)
        # # Trim both arrays to min_len
        true_rate = true_rate_aligned 
        mean_ = simulated_rates_aligned
        print(len(true_rate),len(mean_))
        # mean_ = mean_[:min_len]
        # true_rate = true_rate[:min_len]
        ax.plot(true_rate.index, true_rate, label="Real Data", color="black")
        ax.plot(true_rate.index, mean_, label= name_of_model, color="blue")
        if plot_q:
            # calcute the cut off point 
            lower_q = np.quantile(simulated_rates, downer_quantile/100 , axis = 1)
            upper_q = np.quantile(simulated_rates , upper_quantile/100 , axis = 1)
            #plot quantiles        
            ax.fill_between(true_rate.index, lower_q, upper_q, color="blue", alpha=0.2, 
                        label=f"{int(upper_quantile)}-{int(downer_quantile)}% Quantile Range")
        ax.set_xlabel("Days Simulated")
        ax.set_ylabel("Rate")
        ax.set_title(name_of_model + "vs. Real Data")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax

    def ir_short(self,
                 rate:pd.Series, 
                 start_date: str, 
                 name: Literal['HullWhite_timedependant_nojump', 'HullWhite_timedependant_jump', 'Vasichek','HullWhite_timedependant_nojump_constant_vol'] = 'HullWhite_timedependant_nojump_constant_vol', 
                 simulate_days: int = 252,
                 n_paths = 10000,
                 _plot_graph = False,
                 _plot_q = True):
        """
        This function uses one of the already implemented methods of simulating short_term rate
        ---------------------------------------------------------------------------------------
        In order to simulate/plot the simulation graph this function needs:  
            rate : pd.Series 
                   rates that are needed to simulate 
            start_date: str
                from this date starts the simulation and plotting
            name: str
                 type of functions that will be used to predict the interest rates
            simulate_days: int
                how many days will be simulated using this function  
        Returns:
            (simulated_days, n_paths) of paths in format of pd.DataFrame
        """
        # initliaze the start and end date 
        self.start_date = datetime.strptime(start_date , '%Y-%m-%d')
        self.end_date = self.start_date + relativedelta(days= simulate_days + 1)
        start_window =  self.start_date - pd.Timedelta(days=252)
        end_window =  self.start_date

        # Filter the DataFrame
        train_rates = rate.loc[start_window:end_window]
        test_times = rate.loc[rate.index >= self.start_date].index[:simulate_days]
        test_rate = rate.loc[test_times] 
        
        # due to this cut , the rate could be transferred to functions with a skip in a date
        if name not in self._dispatch_map_short_term:
            raise ValueError(f"Invalid model name '{name}'. Choose from {list(self._dispatch_map_short_term.keys())}")
        
        # choose and fit the function 
        func = self._dispatch_map_short_term[name]

        data = func(train_rates,test_rate,n_paths,simulate_days) # here it returns df of size (simulated_days , n_paths)
        # plot the real data agaisnt the simulated ,
        # the data is given as a dataframe of series of simulated rates 
        if _plot_graph:
            self.draw(test_rate,
                    data,
                    start_date,
                    test_rate.index, 
                    name,
                    upper_quantile= 5,
                    downer_quantile=95,
                    plot_q= False
                    )
        else: 
            return data
    def plot_roc_auc(
                    self, 
                    start_date:str,
                    rate:pd.Series,
                    _return = False,
                    name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                    n_paths =10000,
                    shift_days:int = 1,
                    train_size = 0.8,
                    cv =5 ,
                    top_n:int = 12
                    ,title='ROC AUC Curve'):
        """
        Plot ROC AUC curve for binary classification derived from regression predictions.
        Here we convert regression targets to binary: increase (1) vs decrease/no change (0).
        """
        # Convert true and predicted to binary labels indicating increase vs no increase

        y_pred,y_true = self.short_term_rate.train_predict_cat(start_date=start_date,
                                                                    rate=rate ,
                                                                    _return = _return,
                                                                    name=name,
                                                                    n_paths=n_paths
                                                                    ,shift_days=shift_days,
                                                                    train_size=0.8,
                                                                    cv = cv,
                                                                    top_n=top_n)
        y_true_bin = (y_true.diff().fillna(0) > 0).astype(int)
        y_pred_bin = (pd.Series(y_pred, index=y_true.index).diff().fillna(0) > 0).astype(int)
        auc = roc_auc_score(y_true_bin, y_pred_bin)
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
        
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title + rate.columns[0])
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        
        return auc