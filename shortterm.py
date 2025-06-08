# SHORT TERM
import subprocess
import sys
sys.path.append("..")  # Adjust if notebooks are deeper
import warnings
warnings.filterwarnings("ignore")
# from dataloader import Data 
"""
Decided to skip data importation and 


"""
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
install_package("catboost")
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Union,Literal
from arch import arch_model
from YieldCurve import dataloader
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score, roc_curve
class short_term_rate():
    def __init__(self ,csv_folder_path
                #  ,data_short_term:pd.DataFrame # short term interest rates 
                 ):
        """
        After testing on Russian data i assumed that usually here is interest rates from this list
        3m ,6m, 9m, 1y everything other than that is just spurious , otherwise people wouldn't had any problems 
        """
        self.dataloader_ = dataloader.DataLoader(csv_folder_path)
    def _estimate_params_vasichek(self, rate:pd.Series = None) -> list:
        """
        Euler- Maruama estimation of parameters using mle: 
            fitting AR(1) -> MLE -> params
        """
        model = AutoReg(rate , lags = 1 , old_names = False)
        result = model.fit()
        discretization_method_params = [result.params[0] * 252,(1-result.params[1])*252 , np.sqrt(result.sigma2* 252)]
        return discretization_method_params
    def estimate_vasicek_parameters_discrete(self, rates: pd.Series):
        rates = np.asarray(rates)
        r_lag = rates[:-1]
        r_next = rates[1:]
        dt = 1/252  # or use your actual dt

        # Regression: r_next = phi * r_lag + const
        X = np.column_stack([r_lag, np.ones_like(r_lag)])
        beta, _, _, _ = np.linalg.lstsq(X, r_next, rcond=None)
        phi = beta[0]
        const = beta[1]

        a = -np.log(phi) / dt
        b = const / (1 - phi)
        residuals = r_next - (phi * r_lag + const)
        sigma = np.std(residuals) * np.sqrt(2 * a / (1 - phi**2))
        print(f"Estimated a={a:.4f}, b={b:.4f}, sigma={sigma:.4f}")
        return a, b, sigma


    def vasicek_simulation_paths(self,
        rate:pd.Series,
        theta: float,
        alpha: float,
        sigma: float,
        n_paths: int = 10000,
        simulate_days: int = 252,
        dt: float = 1/252
    ) -> pd.DataFrame:
        """
        Simulate multiple paths of the Vasicek short rate model using Euler-Maruyama.

        Parameters:
            rate: pd.Series (historical rates) or float (initial rate)
            theta: mean reversion speed
            alpha: long-term mean level
            sigma: volatility parameter
            n_paths: number of simulation paths
            simulate_days: number of days to simulate
            dt: time step size (default 1/252 for daily steps)

        Returns:
            pd.DataFrame: Simulated rates, shape (simulate_days, n_paths)
        """
        # Determine initial rate
        if isinstance(rate, float):
            r0 = rate
        elif isinstance(rate, pd.Series):
            r0 = rate.iloc[0]
        else:
            raise ValueError("rate must be a float or pd.Series")

        # Preallocate array for all paths
        estimated = np.zeros((simulate_days, n_paths))
        estimated[0, :] = r0

        # Simulate all paths
        for t in range(1, simulate_days):
            dr = theta * (alpha - estimated[t-1, :]) * dt + sigma * np.sqrt(dt) * np.random.randn(n_paths)
            estimated[t, :] = estimated[t-1, :] + dr

        # Create DataFrame
        df_paths = pd.DataFrame(estimated, columns=[f'Path_{i+1}' for i in range(n_paths)])

        return df_paths
    def theta_from_arima(self,rates, max_p=3, max_d=1, max_q=3, forecast_steps=252):
        """
        Finds the best ARIMA(p,d,q) model (p<max_p+1, d=0 or 1, q<max_q+1) by AIC,
        fits it, and returns a function theta(t) that gives the ARIMA forecast at time t (years) ahead.
        """
        best_aic = np.inf
        best_order = None
        best_fit = None

        # Try all combinations of p, d, q
        for p in range(max_p+1):
            for d in range(max_d+1):
                for q in range(max_q+1):
                    try:
                        model = ARIMA(rates, order=(p, d, q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                            best_fit = fit
                    except Exception:
                        continue

        if best_fit is None:
            raise ValueError("No ARIMA model could be fit.")

        print(f"Best ARIMA order for theta: {best_order}, AIC={best_aic:.2f}")

        # Forecast future mean reversion levels
        forecast = best_fit.forecast(steps=forecast_steps)
        forecast = pd.Series(
            forecast.values,
            index=pd.date_range(rates.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
        )
        t0 = forecast.index[0]

        def theta(t):
            # t in years from forecast start
            date = t0 + pd.Timedelta(days=int(t * 252))
            if date in forecast.index:
                return forecast.loc[date]
            else:
                return forecast.iloc[-1]
        return theta
    def alpha_from_best_arima(self,
    rates, max_p=3, max_d=1, max_q=3, dt=1/252, max_alpha=1.0
):
        """
        Finds the best ARIMA(p,d,q) model (p<max_p+1, d=0 or 1, q<max_q+1) by AIC,
        fits it, and returns a function alpha(t) that gives the mean reversion speed.
        If the best model is not AR(1), falls back to ARIMA(1,0,0).
        Caps alpha at max_alpha.
        """
        import warnings
        best_aic = np.inf
        best_order = None
        best_fit = None

        for p in range(max_p+1):
            for d in range(max_d+1):
                for q in range(max_q+1):
                    try:
                        model = ARIMA(rates, order=(p, d, q))
                        fit = model.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                            best_fit = fit
                    except Exception:
                        continue

        phi = None
        if best_fit is not None and hasattr(best_fit, 'arparams') and len(best_fit.arparams) > 0 and best_order == (1, 0, 0):
            phi = best_fit.arparams[0]
            print(f"Best ARIMA order for alpha: {best_order}, AIC={best_aic:.2f}")
        else:
            # Fallback to ARIMA(1,0,0)
            print("Best ARIMA model is not AR(1), falling back to ARIMA(1,0,0)")
            model = ARIMA(rates, order=(1, 0, 0))
            best_fit = model.fit()
            phi = best_fit.arparams[0]
            print(f"ARIMA(1,0,0) phi: {phi:.4f}")

        # Calculate alpha and cap it
        alpha_value = (1 - phi) / dt
        if alpha_value > max_alpha:
            warnings.warn(f"Alpha ({alpha_value:.4f}) capped at {max_alpha}. AR(1) coefficient phi={phi:.4f}")
            alpha_value = max_alpha
        elif phi < 0.5:
            warnings.warn(f"AR(1) coefficient phi={phi:.4f} is low. Alpha may be unstable.")

        print(f"Final alpha used: {alpha_value:.4f}")

        def alpha(t):
            return alpha_value

        return alpha
    def sigma_from_best_arch(self,rates, forecast_steps=252, returns=True):
        """
        Fits several ARCH/GARCH-type models to the rates (or returns), selects the best by AIC,
        and returns a function sigma(t) for forecasted volatility at t (years ahead).
        Only models supporting multi-step analytic forecasts are considered.
        """
        # if returns:
        #     data = rates.diff().dropna()
        # else:
        #     data = rates.dropna()
        data = rates * 1000
        # List of model specifications to try (only those with analytic multi-step forecast)
        model_specs = [
            {'vol': 'Garch', 'p': 1, 'q': 1, 'power': 2, 'name': 'GARCH(1,1)'},
            {'vol': 'Garch', 'p': 2, 'q': 1, 'power': 2, 'name': 'GARCH(2,1)'},
            {'vol': 'Garch', 'p': 1, 'q': 2, 'power': 2, 'name': 'GARCH(1,2)'},
            {'vol': 'EGarch', 'p': 1, 'q': 1, 'name': 'EGARCH(1,1)'},
            # TGARCH and GJR-GARCH are omitted for multi-step analytic forecast
        ]

        best_aic = np.inf
        best_fit = None
        best_spec = None

        for spec in model_specs:
            try:
                model_kwargs = {k: v for k, v in spec.items() if k in ['vol', 'p', 'o', 'q', 'power']}
                model = arch_model(data, mean='Zero', **model_kwargs)
                fit = model.fit(disp='off')
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
                    best_spec = spec['name']
            except Exception as e:
                continue

        if best_fit is None:
            raise ValueError("No suitable GARCH-type model could be fit.")

        print(f"Best volatility model: {best_spec}, AIC={best_aic:.2f}")

        # Forecast future volatility (variance)
        garch_forecast = best_fit.forecast(horizon=forecast_steps, start=None)
        variance = garch_forecast.variance.values[-1]
        sigma_forecast = np.sqrt(variance)
        forecast_index = pd.date_range(rates.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
        sigma_series = pd.Series(sigma_forecast, index=forecast_index) / 1000

        def sigma(t):
            idx = int(t * 252)
            if idx < len(sigma_series):
                return sigma_series.iloc[idx]
            else:
                return sigma_series.iloc[-1]
        return sigma 

    def Vasichek(self,test_rate:pd.Series,rate:pd.Series,n_paths:int ,simulate_days = 252): # already cut rates 
        params = self.estimate_vasicek_parameters_discrete(test_rate) # oos test
        simulated_date = self.vasicek_simulation_paths(rate,*params,n_paths,simulate_days)
        return simulated_date
    def HullWhite_timedependant_jump(self, 
                                       train_rate:pd.Series,
                                        rate:pd.Series,
                                        n_paths:int,
                                        simulate_days = 252,
                                        random_seed = 42                            
                                        ):
        pass # doesn't work with interest rate as we can't simulate the supporting rate without options
    def HullWhite_timedependant_nojump(self, 
                                       train_rate:pd.Series,
                                        rate:pd.Series,
                                        n_paths:int,
                                        simulate_days = 252,
                                        random_seed = 42                           
                                        ):
        """
        Simulate Hull-White model with function-based parameters.
        Parameters:
            dates: pd.DatetimeIndex for simulation steps
            r0: initial rate
            theta_func, alpha_func, sigma_func: functions of t (years)
            random_seed: for reproducibility
        Returns:
            pd.Series of simulated rates indexed by dates
        """
        #Initilize the values
        theta_func = self.theta_from_arima(train_rate)
        alpha_func = self.alpha_from_best_arima(train_rate)
        sigma_func = self.sigma_from_best_arch(train_rate)

        if random_seed is not None:
            np.random.seed(random_seed)
        dates = rate.index
        start_date = rate.index[0]  # first date in your rate series
        end_date = start_date + pd.DateOffset(years=1)  # one year later

        dates = pd.bdate_range(start=start_date, end=end_date)
        dates = pd.to_datetime(dates)
        N = len(dates)
        dt = np.diff(dates) / np.timedelta64(1, 'D') / 252  # business years
        dt = np.insert(dt, 0, 0)
        r = np.zeros(N)
        r[0] = rate[0]
        t_grid = np.cumsum(dt)  # time in years from start

        for i in range(1, N):
            t = t_grid[i-1]
            dW = np.random.normal(0, np.sqrt(dt[i]))
            drift = theta_func(t) - alpha_func(t) * r[i-1]
            r[i] = r[i-1] + drift * dt[i] + sigma_func(t) * dW

        return pd.Series(r, index=dates, name="Hull-White Simulated Rate")
    
    def features_(self , 
                 start_date:str,
                 rate:pd.Series,
                 _return = False,
                   name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                   n_paths =10000):
        
        self._dispatch_map_short_term = {
            'HullWhite_timedependant_nojump': self.HullWhite_timedependant_nojump,
            'HullWhite_timedependant_jump': self.HullWhite_timedependant_jump,
            'Vasichek': self.Vasichek
        }      
        func = self._dispatch_map_short_term[name]

        simulate_days = 252
        
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif isinstance(start_date, pd.Timestamp):
        # Convert pandas Timestamp to datetime
            self.start_date = start_date.to_pydatetime()
        else:
        # Assume it's already datetime.datetime or compatible
            self.start_date = start_date
        # self.start_date = datetime.strptime(self.start_date , '%Y-%m-%d')
        self.end_date = self.start_date + relativedelta(days= simulate_days + 1)
        start_window =  self.start_date - pd.Timedelta(days=252)
        end_window =  self.start_date

        # Filter the DataFrame
        train_rates = rate.loc[start_window:end_window]
        test_times = rate.loc[rate.index >= self.start_date].index[:simulate_days]
        test_rate = rate.loc[test_times] 

        sde_simulated = func(train_rates,test_rate,n_paths,simulate_days) 
        inflation_data = self.dataloader_.inflation(True)
        inflation_data = self.dataloader_.extract_all_features(inflation_data , _return = True)
        key_rate_features = self.dataloader_.build_final_feature_dataframe(self.dataloader_.changes_in_key_rate(True))
        merged_df = pd.concat([sde_simulated, inflation_data, key_rate_features], axis=1)
        
        self.merged_df = merged_df.dropna()

        if _return:
            return self.merged_df
    def align_target(self, features_df, target_series, shift_days=1):
        """
        Align target to be the value of target_series shifted backward by shift_days,
        so features at time t correspond to target at time t + shift_days.
        
        Parameters:
        - features_df: pd.DataFrame with datetime index of features
        - target_series: pd.Series with datetime index of target variable (e.g., interest rate)
        - shift_days: int, number of days to shift target backward (default 1 for tomorrow)
        
        Returns:
        - features_df aligned with target (rows with matching dates)
        - target_aligned: pd.Series with target shifted and aligned to features
        """
        target_aligned = target_series.shift(-shift_days)
        # Align indexes by intersection (drop rows with NaN in target or features)
        combined_index = features_df.index.intersection(target_aligned.dropna().index)
        features_aligned = features_df.loc[combined_index]
        target_aligned = target_aligned.loc[combined_index]
        return features_aligned, target_aligned
    def time_series_train_test_split(self, features, target, train_size=0.8):
        """
        Split features and target into train and test sets respecting time order.
        
        Parameters:
        - features: pd.DataFrame of features
        - target: pd.Series of target variable
        - train_size: float between 0 and 1, fraction of data to use for training
        
        Returns:
        - X_train, X_test, y_train, y_test
        """
        n = len(features)
        train_end = int(n * train_size)
        X_train = features.iloc[:train_end]
        X_test = features.iloc[train_end:]
        y_train = target.iloc[:train_end]
        y_test = target.iloc[train_end:]
        return X_train, X_test, y_train, y_test
    def lasso_reg_(self,X_train, y_train, cv=5, random_state=42, top_n=None):
        """
        Fit Lasso regression with cross-validation and return list of selected features.
        Optionally select the top N features by absolute coefficient magnitude.
        
        Parameters:
        - X_train: pd.DataFrame of training features
        - y_train: pd.Series of training target
        - cv: int, number of folds for cross-validation
        - random_state: int, random seed
        - top_n: int or None, number of top features to select by coefficient magnitude.
                If None, all non-zero coefficient features are returned.
        
        Returns:
        - selected_features: list of feature names selected
        - model: fitted LassoCV model
        """
        # Fit LassoCV
        
        lasso = LassoCV(cv=cv, random_state=random_state).fit(X_train, y_train)
        
        # Get absolute coefficients
        coef_abs = np.abs(lasso.coef_)
        
        # Get indices of features with non-zero coefficients
        nonzero_idx = np.where(coef_abs > 0)[0]
        
        if top_n is not None and top_n > 0:
            # Sort non-zero coefficients by magnitude descending
            sorted_idx = nonzero_idx[np.argsort(coef_abs[nonzero_idx])[::-1]]
            # Select top_n features or fewer if not enough
            selected_idx = sorted_idx[:top_n]
        else:
            # Select all non-zero features
            selected_idx = nonzero_idx
        
        selected_features = X_train.columns[selected_idx].tolist()
        
        return selected_features, lasso
    def selected_features_lasso(self,start_date:str,
                        rate:pd.Series,
                        _return = False,
                        name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                        n_paths =10000,
                        shift_days:int = 1,
                        train_size = 0.8,
                        cv =5 ,
                        top_n:int = 12
                        ):
        """
        Create a DataFrame of selected features and their importance (absolute coefficients).
        
        Parameters:
        - X_train: pd.DataFrame of training features (used to map feature names to coefficients)
        - lasso_model: fitted LassoCV model (already trained, no refitting)
        - selected_features: list of selected feature names
        
        Returns:
        - pd.DataFrame with columns ['Feature', 'Importance'], sorted descending by importance
        """
        merged_df_features = self.features_(start_date=start_date, 
                       rate=rate,
                       _return= True,
                       name='HullWhite_timedependant_nojump', 
                       n_paths= 10000
                       )
        features_aligned, target_aligned = self.align_target(merged_df_features,rate,shift_days=shift_days)
        X_train, X_test, y_train, y_test = self.time_series_train_test_split(features_aligned, target_aligned, train_size=train_size)
        selected_features, lasso = self.lasso_reg_(y_train=y_train,X_train=X_train, cv= cv , top_n= 100)
        coefs = lasso.coef_
        print(type(coefs))
        print(coefs.shape if hasattr(coefs, 'shape') else None)
        print(coefs)
        importance = {feat: abs(coefs[X_train.columns.get_loc(feat)]) for feat in selected_features}
        
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        # importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df
    def deez_on_mya_buts(self):
        return 'fart'
    def worker_thingy(self,start_date:str,
                        rate:pd.Series,
                        _return = False,
                        name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                        n_paths =10000,
                        shift_days:int = 1,
                        train_size = 0.8,
                        cv =5 ,
                        top_n:int = 12
                        ):
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.start_date = start_date  # already datetime object
        merged_df_features = self.features_(start_date=start_date, 
                       rate=rate,
                       _return= True,
                       name='HullWhite_timedependant_nojump', 
                       n_paths= 10000
                       )
        features_aligned, target_aligned = self.align_target(merged_df_features,rate,shift_days=shift_days)
        X_train, X_test, y_train, y_test = self.time_series_train_test_split(features_aligned, target_aligned, train_size=train_size)
        
        selected_features, lasso = self.lasso_reg_(y_train=y_train,X_train=X_train, cv= cv , top_n= top_n)

        X_train_lasso = X_train[selected_features]
        X_test_lasso = X_test[selected_features]

        return X_train_lasso, X_test_lasso, y_train, y_test
    def yeo_johnson_transform(self, X_train, X_test, y_train, y_test):
        """
        Apply Yeo-Johnson transformation to features and target.
        """
        pt_X = PowerTransformer(method='yeo-johnson', standardize=True)
        pt_y = PowerTransformer(method='yeo-johnson', standardize=True)
        
        X_train_t = pd.DataFrame(pt_X.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_t = pd.DataFrame(pt_X.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        y_train_t = pd.Series(pt_y.fit_transform(y_train.values.reshape(-1,1)).flatten(), index=y_train.index)
        y_test_t = pd.Series(pt_y.transform(y_test.values.reshape(-1,1)).flatten(), index=y_test.index)
        
        return X_train_t, X_test_t, y_train_t, y_test_t, pt_y
    def time_series_cv_ridge(self, X_train, y_train, cv_splits=5):
        """
        Best coefficinetns using ridge regression
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        model = Ridge(random_state=42)
        
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV MSE: {-grid_search.best_score_}")
        
        return grid_search.best_estimator_
    def time_series_cv_catboost(self, X_train, y_train, cv_splits=5):
        """
        Perform time-series cross-validation with CatBoost and hyperparameter tuning.
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        model = CatBoostRegressor(verbose=0, random_seed=42)
        
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200, 300]
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV MSE: {-grid_search.best_score_}")
        
        return grid_search.best_estimator_
    def train_predict_cat(self,start_date:str,
                        rate:pd.Series,
                        _return = False,
                        name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                        n_paths =10000,
                        shift_days:int = 1,
                        train_size = 0.8,
                        cv =5 ,
                        top_n:int = 12
                        ):
        X_train_lasso, X_test_lasso, y_train, y_test = self.worker_thingy(start_date=start_date,
                                                                          rate=rate,
                                                                          _return = _return
                                                                          ,name=name,
                                                                          n_paths=n_paths,
                                                                          shift_days=shift_days,
                                                                          train_size= train_size,
                                                                          cv = cv,
                                                                          top_n=top_n)
        X_train_t, X_test_t, y_train_t, y_test_t, pt_y = self.yeo_johnson_transform(X_train_lasso, X_test_lasso, y_train, y_test)

        best_estimator_ = self.time_series_cv_catboost(X_train_t,y_train_t)
        y_pred_t = best_estimator_.predict(X_test_t)

        #  Inverse transform predictions to original scale
        y_pred = pd.Series(pt_y.inverse_transform(y_pred_t.reshape(-1,1)).flatten(), index=y_test_t.index)
        
        return y_pred, y_test
    def train_predict_linear(self,start_date:str,
                        rate:pd.Series,
                        _return = False,
                        name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                        n_paths =10000,
                        shift_days:int = 1,
                        train_size = 0.8,
                        cv =5 ,
                        top_n:int = 12
                        ):
        
        X_train_lasso, X_test_lasso, y_train, y_test = self.worker_thingy(start_date=start_date,
                                                                          rate=rate,
                                                                          _return = _return
                                                                          ,name=name,
                                                                          n_paths=n_paths,
                                                                          shift_days=shift_days,
                                                                          train_size= train_size,
                                                                          cv = cv,
                                                                          top_n=top_n)
        
        X_train_t, X_test_t, y_train_t, y_test_t, pt_y = self.yeo_johnson_transform(X_train_lasso, X_test_lasso, y_train, y_test)

        best_estimator_ = self.time_series_cv_ridge(X_train_t,y_train_t)
        y_pred_t = best_estimator_.predict(X_test_t)

        #  Inverse transform predictions to original scale
        y_pred = pd.Series(pt_y.inverse_transform(y_pred_t.reshape(-1,1)).flatten(), index=y_test_t.index)
        
        # return y_pred_t,y_pred
        return y_pred, y_test
    def train_predict_model(self,start_date:str,
                        rate:pd.Series,
                        _return = False,
                        name:Literal['HW_constant_nojump', 'HW_timedependant_jump', 'Vasichek'] = 'HullWhite_timedependant_nojump',
                        n_paths =10000,
                        shift_days:int = 1,
                        train_size = 0.8,
                        cv =5 ,
                        top_n:int = 12
                        ):
        
        X_train_lasso, X_test_lasso, y_train, y_test = self.worker_thingy(start_date=start_date,
                                                                          rate=rate,
                                                                          _return = _return
                                                                          ,name=name,
                                                                          n_paths=n_paths,
                                                                          shift_days=shift_days,
                                                                          train_size= train_size,
                                                                          cv = cv,
                                                                          top_n=top_n)
        
        X_train_t, X_test_t, y_train_t, y_test_t, pt_y = self.yeo_johnson_transform(X_train_lasso, X_test_lasso, y_train, y_test)

        best_estimator_ = self.time_series_cv_ridge(X_train_t,y_train_t)
        y_pred_t = best_estimator_.predict(X_test_t)
        feature_names = X_train_lasso.columns

        coefs = best_estimator_.coef_
        if coefs.ndim > 1:
            # If multi-target, average absolute coefficients
            coefs = np.mean(np.abs(coefs), axis=0)
        
        # Add constant for intercept in statsmodels
        X_sm = sm.add_constant(X_train_lasso)
        
        # Fit OLS model using statsmodels to get p-values
        model_sm = sm.OLS(y_train, X_sm).fit()
        
        # Extract p-values (first is intercept)
        pvalues = model_sm.pvalues[1:]  # exclude intercept
        
        # Build DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'p-value': pvalues.values
        })
        
        # Sort by absolute coefficient descending
        df['abs_coef'] = df['Coefficient'].abs()
        df = df.sort_values(by='abs_coef', ascending=False).drop(columns='abs_coef').reset_index(drop=True)
        
        return df
 

    def estimate_constant_sigma(self, r: pd.Series, alpha_func, theta_func):
        """
        Estimate constant sigma from short rate data r, given alpha and theta functions.

        Parameters:
            r: pd.Series of observed short rates indexed by dates
            alpha_func: function alpha(t) mean reversion speed
            theta_func: function theta(t) mean reversion level

        Returns:
            float: estimated constant sigma
        """
        dates = r.index
        # Calculate dt in years between observations (business days / 252)
        dt = np.diff(dates) / np.timedelta64(1, 'D') / 252
        residuals = []
        for i in range(1, len(r)):
            t = (dates[i-1] - dates[0]).days / 252  # time in years from start
            drift = alpha_func(t) * (theta_func(t) - r.iloc[i-1]) * dt[i-1]
            increment = r.iloc[i] - r.iloc[i-1]
            residuals.append(increment - drift)
        residuals = np.array(residuals)
        sigma_hat = np.sqrt(np.sum(residuals**2) / (len(residuals) * np.mean(dt)))
        return sigma_hat

    def HullWhite_timedependant_nojump_constant_vol(self, 
                                   train_rate: pd.Series,
                                   rate: pd.Series,
                                   n_paths: int,
                                   simulate_days=252,
                                   random_seed=42,
                                   use_constant_sigma=True):
        """
        Simulate Hull-White model with function-based parameters.
        Parameters:
            train_rate: pd.Series of historical rates for parameter estimation
            rate: pd.Series of initial rates for simulation start
            n_paths: number of Monte Carlo paths (currently single path simulated)
            simulate_days: number of business days to simulate
            random_seed: for reproducibility
            use_constant_sigma: if True, estimate and use constant sigma
        Returns:
            pd.Series of simulated rates indexed by dates
        """
        # Initialize parameter functions
        theta_func = self.theta_from_arima(train_rate)
        alpha_func = self.alpha_from_best_arima(train_rate)
        sigma_func = self.sigma_from_best_arch(train_rate)

        if use_constant_sigma:
            sigma_const = self.estimate_constant_sigma(train_rate, alpha_func, theta_func)
            sigma_func = lambda t: sigma_const  # override sigma_func with constant sigma

        if random_seed is not None:
            np.random.seed(random_seed)

        start_date = rate.index[0]
        end_date = start_date + pd.DateOffset(years=1)
        dates = pd.bdate_range(start=start_date, end=end_date)
        dates = pd.to_datetime(dates)
        N = len(dates)

        dt = np.diff(dates) / np.timedelta64(1, 'D') / 252  # business years
        dt = np.insert(dt, 0, 0)
        r = np.zeros(N)
        r[0] = rate.iloc[0]
        t_grid = np.cumsum(dt)  # time in years from start

        for i in range(1, N):
            t = t_grid[i-1]
            dW = np.random.normal(0, np.sqrt(dt[i]))
            drift = theta_func(t) - alpha_func(t) * r[i-1]
            r[i] = r[i-1] + drift * dt[i] + sigma_func(t) * dW

        return pd.Series(r, index=dates, name="Hull-White Simulated Rate")

        

    