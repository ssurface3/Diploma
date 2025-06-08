# grave here
# Fit ARIMA model to a parameter time series and return a forecasting function
    def fit_arima_parameter_with_aic_search(self, time_series, p_range=(0,3), d=0, q_range=(0,3)):
        """
        Fit ARIMA model to a time series by searching over combinations of p and q,
        selecting the model with the lowest AIC.

        Inputs:
        - time_series (array-like): Historical parameter values indexed by time.
        - p_range (tuple): (min_p, max_p) inclusive range for AR order p.
        - d (int): Differencing order.
        - q_range (tuple): (min_q, max_q) inclusive range for MA order q.

        Outputs:
        - best_model: Fitted ARIMAResults object with lowest AIC.
        - param_func (function): Function t -> forecasted parameter value at time t.
        """
        best_aic = np.inf
        best_order = None
        best_model = None

        for p in range(p_range[0], p_range[1]+1):
            for q in range(q_range[0], q_range[1]+1):
                try:
                    model = ARIMA(time_series, order=(p,d,q))
                    fitted = model.fit()
                    aic = fitted.aic
                    # Uncomment for debugging:
                    # print(f"Tested ARIMA({p},{d},{q}) - AIC: {aic:.2f}")
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p,d,q)
                        best_model = fitted
                except Exception:
                    # Model failed to fit, skip silently or log if needed
                    continue

        if best_model is None:
            raise ValueError("No ARIMA model could be fit to the data.")

        print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")

        def param_func(self, t):
            """
            Forecast parameter at time t.

            Inputs:
            - t (int or float): Time index at which to forecast parameter.

            Outputs:
            - float: Forecasted parameter value at time t.
            """
            n = len(time_series)
            if t < n:
                return time_series[int(t)]
            else:
                steps_ahead = int(t - (n - 1))
                forecast = best_model.get_forecast(steps=steps_ahead)
                return forecast.predicted_mean.iloc[-1]

        return best_model, param_func


    # Estimate constant Hull-White parameters (a, theta, sigma) from short rate data
    # def estimate_hw_parameters(self, short_rates, dt=1.0): # they are constant , just for testing
    #     """
    #     Estimate mean reversion speed 'a', long-term mean 'theta', and volatility 'sigma'
    #     from observed short rate time series using discretized Vasicek approximation. 
    #     Returns constant parameters
    #     ---------------------------------------------------------------------------------
    #     Inputs:
    #     - short_rates (array-like): Observed short rate values over time.
    #     - dt (float): Time step size between observations (default 1.0).

    #     Outputs:
    #     - a_est (float): Estimated mean reversion speed.
    #     - theta_est (float): Estimated long-term mean level.
    #     - sigma_est (float): Estimated volatility of diffusion.
    #     """
    #     r = np.asarray(short_rates)
    #     dr = np.diff(r)
    #     r_lag = r[:-1]

    #     # Linear regression of dr/dt on r_t to estimate parameters
    #     reg = LinearRegression().fit(r_lag.reshape(-1,1), dr/dt)
    #     a_est = -reg.coef_[0]
    #     theta_est = reg.intercept_ / a_est if a_est != 0 else 0

    #     residuals = dr/dt - reg.predict(r_lag.reshape(-1,1))
    #     sigma_est = np.std(residuals) * np.sqrt(dt)

    #     return a_est, theta_est, sigma_est


    # Simulate Hull-White short rate paths with time-dependent parameters
    # def simulate_hull_white(self, r0, a_func, theta_func, sigma_func, times, n_paths=1, seed=None, return_mean=False): # returns paths 
    #     """
    #     Simulate Hull-White short rate paths with time-dependent parameters.
    #     -------------------------------------------------------------------
    #     Inputs:
    #     - r0 (float): Initial short rate.
    #     - a_func (function): Returns mean reversion speed a(t).
    #     - theta_func (function): Returns mean reversion level theta(t).
    #     - sigma_func (function): Returns volatility sigma(t).
    #     - times (array-like): Time grid for simulation.
    #     - n_paths (int): Number of simulation paths.
    #     - seed (int or None): Random seed.
    #     - return_mean (bool): If True, return average path across all simulations.

    #     Outputs:
    #     - pd.DataFrame of shape (len(times), n_paths) if return_mean=False. # for cool graph that will show quantiles
    #     - pd.Series of length len(times) if return_mean=True. # for interest rate estimation
    #     """
    #     if seed is not None:
    #         np.random.seed(seed)
    #     # Convert timestamps to days since epoch
    #     times_numeric = (times - pd.Timestamp("1970-01-01")).days.values  # shape (n_steps,)
    #     print('hui')
    #     dt = np.diff(times_numeric, prepend=times_numeric[0])
    #     n_steps = len(times)
    #     print(times_numeric, times_numeric[0])
    #     dt = np.diff(times_numeric, prepend=times_numeric[0])  # <-- FIXED HERE
    #     dt_years = dt / 365.0  # Convert days to years
    #     rates = np.zeros((n_steps, n_paths))
    #     rates[0, :] = r0

    #     for i in range(1, n_steps):
    #         t_prev_days = times_numeric[i-1]
    #         dt_i = dt_years[i]

    #         # Handle constant or time-dependent parameters
    #         a_t = a_func(t_prev_days) if callable(a_func) else a_func
    #         theta_t = theta_func(t_prev_days) if callable(theta_func) else theta_func
    #         sigma_t = sigma_func(t_prev_days) if callable(sigma_func) else sigma_func

    #         dW = np.random.normal(size=n_paths) * np.sqrt(dt_i)
    #         dr = a_t * (theta_t - rates[i-1, :]) * dt_i + sigma_t * dW
    #         rates[i, :] = rates[i-1, :] + dr

    #     df_rates = pd.DataFrame(rates, index=times)

    #     if return_mean:
    #         return df_rates.mean(axis=1)
    #     else:
            # return df_rates
    def HW_constant_nojump(self , rate:Union[pd.Series,float] , start_date:str = None, simulate_days:int = 252, n_paths:int= 1000) -> pd.Series:
        """
        Autonomous function that takes the first rate and predicts for simulate_days days
            Model:Uses basic constant parameter Hull-WHite as a bakcbone 
        ---------------------------------------------------------------------------------
        Output: 
            simulate_days of output  
        """    
        self.n_paths  = n_paths
        self.start_date = datetime.strptime(start_date , '%Y-%m-%d')
        self.end_date = self.start_date + relativedelta(days= simulate_days + 1)

        if isinstance(rate, pd.Series):
            r0 = rate[0] # initial rate to start calculation 
        elif isinstance(rate, float):
            r0 = rate
        else:
            raise TypeError('The rate is not float or pd.Series, please correct the type')
        
        if len(rate) == simulate_days:
            a_func_ , theta_func_, sigma_func_ = self.estimate_hw_parameters(rate) # constant parameters that don't change over time
        else: # if not filtered -> filter
            rate = rate[(rate.index >= self.start_date)& (rate.index < self.end_date)]
            a_func_ , theta_func_, sigma_func_ = self.estimate_hw_parameters(rate)

        times_  = rate[(rate.index >= self.start_date)& (rate.index < self.end_date)].index # returns dates so the graph is clean 
        self.n_paths = n_paths

        return self.simulate_hull_white(r0 
                                        
                                        ,a_func_ , 
                                        theta_func_, 
                                        sigma_func_ , 

                                        times_ , 
                                        self.n_paths, 
                                        seed = 42 , 
                                        return_mean=False # in order to plot the quantiles
                                )
    def HW_timedependant_jump(self):
        
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# --- Standalone parameter functions ---

# def theta_from_arima(rates, arima_order=(1,0,0), forecast_steps=252):
#     """
#     Returns a function theta(t) that gives the ARIMA forecast at time t (in years) from the end of rates.
#     rates: pd.Series indexed by date
#     arima_order: ARIMA order tuple (p,d,q)
#     forecast_steps: number of steps to forecast (e.g., 252 for 1 year of business days)
#     """
#     # Fit ARIMA model to historical rates
#     model = ARIMA(rates, order=arima_order)
#     fit = model.fit()
#     # Forecast future mean reversion levels
#     forecast = fit.forecast(steps=forecast_steps)
#     forecast = pd.Series(forecast.values, 
#                          index=pd.date_range(rates.index[-1] + pd.Timedelta(days=1), 
#                                              periods=forecast_steps, freq='B'))
#     t0 = forecast.index[0]
#     def theta(t):
#         # Convert t (years) to a date
#         date = t0 + pd.Timedelta(days=int(t * 252))  # 252 business days per year
#         # Find closest date in index
#         if date in forecast.index:
#             return forecast.loc[date]
#         else:
#             return forecast.iloc[-1]  # fallback to last value
#     return theta
def theta_from_arima(rates, max_p=3, max_d=1, max_q=3, forecast_steps=252):
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

# def alpha_from_arima(
#     rates,
#     sim_dates,
#     arima_orders=[(1,0,0), (2,0,0), (1,0,1), (2,0,1), (1,1,0)],
#     window=100
# ):
#     """
#     Returns a pd.Series alpha indexed by sim_dates, where each alpha is the AR(1) coefficient
#     from the best ARIMA model (by AIC) fitted on a rolling window.
    
#     rates: pd.Series of historical rates (indexed by date)
#     sim_dates: pd.DatetimeIndex for simulation
#     arima_orders: list of ARIMA orders to try
#     window: rolling window size (in days)
#     """
#     alphas = []
#     for i in range(len(sim_dates)):
#         # Find the window in historical data ending at sim_dates[i]
#         end_date = sim_dates[i]
#         start_date = end_date - pd.Timedelta(days=window)
#         window_rates = rates[(rates.index > start_date) & (rates.index <= end_date)]
#         if len(window_rates) < 10:  # not enough data
#             alphas.append(np.nan)
#             continue
#         best_aic = np.inf
#         best_model = None
#         # Try all ARIMA orders
#         for order in arima_orders:
#             try:
#                 model = ARIMA(window_rates, order=order)
#                 fit = model.fit()
#                 if fit.aic < best_aic:
#                     best_aic = fit.aic
#                     best_model = fit
#             except Exception:
#                 continue
#         # Extract AR(1) coefficient as 'alpha'
#         if best_model is not None and hasattr(best_model, 'arparams') and len(best_model.arparams) > 0:
#             phi = best_model.arparams[0]
#             dt = 1/252  # assuming daily business steps
#             alpha = (1 - phi) / dt
#         else:
#             alpha = np.nan
#         alphas.append(alpha)
#     return pd.Series(alphas, index=sim_dates, name='alpha')
# def alpha_from_arima_function(rates, arima_order=(1,0,0), forecast_steps=252):
#     """
#     Returns a function alpha(t) that gives the ARIMA-based mean reversion speed
#     at time t (in years) from the end of rates.
#     """
#     # Fit ARIMA model to historical rates
#     model = ARIMA(rates, order=arima_order)
#     fit = model.fit()
#     # Forecast AR(1) coefficients (phi) for the forecast_steps
#     # For ARIMA(1,0,0), phi is constant, so we use the fitted parameter
#     if hasattr(fit, 'arparams') and len(fit.arparams) > 0:
#         phi = fit.arparams[0]
#     else:
#         phi = 0.0  # fallback if AR param is not available
#     dt = 1/252  # assuming daily business steps
#     alpha_value = (1 - phi) / dt

#     def alpha(t):
#         # t: time in years from the end of rates
#         # For ARIMA(1,0,0), alpha is constant; for more complex models, you could extend this
#         return alpha_value

#     return alpha
def alpha_from_arima_function(rates, max_p=3, max_d=1, max_q=3):
    """
    Finds the best ARIMA(p,d,q) model (p<max_p+1, d=0 or 1, q<max_q+1) by AIC,
    fits it, and returns a function alpha(t) that gives the mean reversion speed
    (constant for ARIMA(1,0,0), can be extended for more complex models).
    """
    best_aic = np.inf
    best_order = None
    best_fit = None

    for p in range(1,max_p+1):
        for d in range(max_d+1):
            for q in range(0,max_q+1):
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

    print(f"Best ARIMA order for alpha: {best_order}, AIC={best_aic:.2f}")

    # For ARIMA(1,0,0), phi is the AR(1) coefficient
    if hasattr(best_fit, 'arparams') and len(best_fit.arparams) > 0:
        phi = best_fit.arparams[0]
    else:
        phi = 0.0
    dt = 1/252
    alpha_value = (1 - phi) / dt

    def alpha(t):
        # For ARIMA(1,0,0), alpha is constant
        return alpha_value

    return alpha


import pandas as pd
from arch import arch_model

# def sigma_from_garch(rates, returns=True):
#     """
#     Estimate time-varying volatility (sigma_t) using a GARCH(1,1) model.
    
#     Parameters:
#         rates: pd.Series of rates (indexed by date)
#         returns: if True, use rate differences as input to GARCH
        
#     Returns:
#         pd.Series of conditional sigmas (volatility), indexed by date (aligned to input)
#     """
#     if returns:
#         # Use daily changes (returns) for GARCH modeling
#         data = rates.diff().dropna()
#     else:
#         data = rates.dropna()
    
#     # Fit GARCH(1,1) model
#     model = arch_model(data, vol='Garch', p=1, q=1, mean='Zero')
#     fit = model.fit(disp='off')
    
#     # Get conditional volatility (sigma_t)
#     sigma_series = fit.conditional_volatility
#     sigma_series.name = 'sigma_t'
    
#     # Align index to original dates (if using returns, shift index by 1)
#     if returns:
#         sigma_series.index = rates.index[1:]
#     else:
#         sigma_series.index = rates.index
    
#     return sigma_series
# def sigma_from_garch_function(rates, forecast_steps=252, returns=True):
#     """
#     Returns a function sigma(t) that gives the GARCH(1,1) forecasted volatility
#     at time t (in years) from the end of rates.

#     Parameters:
#         rates: pd.Series of rates (indexed by date)
#         forecast_steps: number of steps to forecast (e.g., 252 for 1 year)
#         returns: if True, use rate differences as input to GARCH

#     Returns:
#         sigma(t): function returning volatility at t (years ahead)
#     """
#     if returns:
#         data = rates.diff().dropna()
#     else:
#         data = rates.dropna()

#     # Fit GARCH(1,1) model
#     model = arch_model(data * 1000, vol='Garch', p=1, q=1, mean='Zero')
#     fit = model.fit(disp='off')

#     # Forecast future volatility (variance)
#     garch_forecast = fit.forecast(horizon=forecast_steps, start=None)
#     # Get the forecasted variance for the next forecast_steps
#     variance = garch_forecast.variance.values[-1]
#     sigma_forecast = np.sqrt(variance)
#     # Create a pd.Series indexed by forecast dates
#     forecast_index = pd.date_range(rates.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
#     sigma_series = pd.Series(sigma_forecast, index=forecast_index) / 1000

#     def sigma(t):
#         # t: time in years from the start of forecast
#         idx = int(t * 252)  # 252 business days per year
#         if idx < len(sigma_series):
#             return sigma_series.iloc[idx]
#         else:
#             return sigma_series.iloc[-1]  # fallback to last value

#     return sigma

# 
def sigma_from_best_arch(rates, forecast_steps=252, returns=True):
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

# def simulate_hull_white_no_jump(r0, theta_series, alpha_series, sigma_series, sim_dates, random_seed=None):
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     r = np.zeros(len(sim_dates))
#     r[0] = r0
#     for i in range(1, len(sim_dates)):
#         dt = (sim_dates[i] - sim_dates[i-1]).days / 252  # business years
#         if np.isnan(alpha_series.iloc[i]) or np.isnan(theta_series.iloc[i]) or np.isnan(sigma_series.iloc[i]):
#             r[i] = r[i-1]
#             continue
#         dW = np.random.normal(0, np.sqrt(dt))
#         drift = theta_series.iloc[i] - alpha_series.iloc[i] * r[i-1]
#         r[i] = r[i-1] + drift * dt + sigma_series.iloc[i] * dW
#     return pd.Series(r, index=sim_dates, name="Hull-White Simulated Rate")
def alpha_from_best_arima(
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
# def hull_white_no_jump_simulation(
#     rates,
#     sim_dates,
#     garch_returns=True,
#     random_seed=None
# ):
#     theta_series = theta_from_arima(rates, sim_dates)
#     alpha_series = alpha_from_arima(rates, sim_dates)
#     sigma_series = sigma_from_garch(rates, sim_dates, returns=garch_returns)
#     r0 = rates.iloc[-1]
#     return simulate_hull_white_no_jump(r0, theta_series, alpha_series, sigma_series, sim_dates, random_seed=random_seed)

def simulate_hull_white_no_jump_functional(
    dates,
    r0,
    theta_func,
    alpha_func,
    sigma_func,
    random_seed=None
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
    if random_seed is not None:
        np.random.seed(random_seed)
    dates = pd.to_datetime(dates)
    N = len(dates)
    dt = np.diff(dates) / np.timedelta64(1, 'D') / 252  # business years
    dt = np.insert(dt, 0, 0)
    r = np.zeros(N)
    r[0] = r0
    t_grid = np.cumsum(dt)  # time in years from start

    for i in range(1, N):
        t = t_grid[i-1]
        dW = np.random.normal(0, np.sqrt(dt[i]))
        drift = theta_func(t) - alpha_func(t) * r[i-1]
        r[i] = r[i-1] + drift * dt[i] + sigma_func(t) * dW

    return pd.Series(r, index=dates, name="Hull-White Simulated Rate")
# # --- Wrapper function ---
# def HullWhite_timedependant_nojump(
#     rates,
#     garch_returns=True,
#     random_seed=None
# ):
#     sim_dates = rates.index
#     theta_series = theta_from_arima(rates, sim_dates)
#     alpha_series = alpha_from_arima_function(rates, sim_dates)
#     sigma_series = sigma_from_garch_function(rates, sim_dates, returns=garch_returns)
#     # r0 = rates.iloc[-1]
#     r0 = rates.iloc[0]
#     return simulate_hull_white_no_jump_functional(r0, theta_series, alpha_series, sigma_series, sim_dates, random_seed=random_seed)

# --- Example usage ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate for 1 year, daily steps
    rate = st['3m'][(st['3m'].index > '2019-01-01')& (st['3m'].index < '2020-01-01')]
    r0 = rate.iloc[0]
    dates = rate.index
    rates_simulated= simulate_hull_white_no_jump_functional(r0=rate.iloc[0],
        dates=dates , theta_func=theta_from_arima(rate) , 
        alpha_func=alpha_from_best_arima(rate),
        sigma_func=sigma_from_best_arch(rate),
        random_seed=42
    )  
    # dates,
    # r0,
    # theta_func,
    # alpha_func,
    # sigma_func,
    # random_seed=None
    # rates_simulated.plot(title="Hull-White (Time-Dependent, No Jump)")
    plt.plot(dates,rates_simulated)
    plt.plot(dates, rate)
    plt.ylabel("Short Rate")
    plt.show()