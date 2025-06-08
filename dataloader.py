# imports
import pandas as pd
import os
import requests
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm  # for progress tracking
import ruptures as rpt
import matplotlib.pyplot as plt
import ruptures as rt
from typing import Union
import re
from statsmodels.tsa.stattools import coint
from itertools import combinations
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict


class DataLoader():
    def __init__(self,csv_folder_path) -> None:
        self.csv_folder_path = csv_folder_path
    def combine_bond_yield_csvs(self,_return = False) -> pd.DataFrame:
        """
        Combine multiple bond yield CSV files into one DataFrame with:
        - One row per unique tradedate
        - Columns for each bond period's yield
        
        Args:
            csv_folder_path (str): Path to folder containing the CSV files
            
        Returns:
            pd.DataFrame: Combined DataFrame with tradedate as index and yields as columns
        """
        # Initialize empty DataFrame to store results
        combined_df = pd.DataFrame()
        
        # Loop through each CSV file in the folder
        for filename in os.listdir(self.csv_folder_path):
            if filename.endswith('.csv'):
                # Extract the bond period from filename (e.g., '3m' from '3m.csv')
                period = filename.split('.')[0]
                
                # Read the CSV file
                df = pd.read_csv(os.path.join(self.csv_folder_path, filename),delimiter=';', skiprows=2)
                
                # Clean the data:
                # 1. Convert comma decimals to dots
                df[df.columns[2]] = df[df.columns[2]].str.replace(',', '.').astype(float)
                
                # 2. Keep only tradedate and yield column
                temp_df = df[['tradedate', df.columns[2]]].copy()
                temp_df.rename(columns={df.columns[2]: period}, inplace=True)
                
                # Merge with combined_df
                if combined_df.empty:
                    combined_df = temp_df
                else:
                    combined_df = pd.merge(combined_df, temp_df, on='tradedate', how='outer')
        
        # Convert tradedate to datetime and sort
        combined_df['tradedate'] = pd.to_datetime(combined_df['tradedate'], format='%d.%m.%Y')
        combined_df.sort_values('tradedate', inplace=True)
        combined_df.set_index('tradedate', inplace=True)

        self.combined_df = combined_df / 100 # because they are in kind of 8.5 not 0.085 

        if _return:
            return combined_df # we return the dataframe with all the needed data
    def __dateframe(self):
        """
        saves start_date and end_date using a dataframe 
        -------------------------------------------------
        """
      
        self.start_date = self.combined_df.index[0]
        self.end_date = self.combined_df.index[-1]

    def moex_data(self,ticker:str = 'SBER', _return = False) -> pd.DataFrame:
        """
        Function that returns prices of the chosen stock 
        ------------------------------------------------
        1. Handles date gaps correctly
        2. Removes nulls properly
        3. Maintains continuous date range
        4. Tracks failed downloads
        ------------------------------------------------
        if _returns = True -> returns a pd.DataFrame
        """
        base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
        
        # Convert to datetime and create full date index
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        all_dates = pd.date_range(start_dt, end_dt, freq='D')
        final_df = pd.DataFrame(index=all_dates)
        
        # Download in 1-month chunks with error handling
        chunk_start = start_dt
        failures = []
        
        with tqdm(total=(end_dt - start_dt).days, desc=f"Downloading {ticker}") as pbar:
            while chunk_start <= end_dt:
                chunk_end = min(chunk_start + timedelta(days=30), end_dt)
                
                params = {
                    'from': chunk_start.strftime('%Y-%m-%d'),
                    'till': chunk_end.strftime('%Y-%m-%d'),
                    'iss.meta': 'off',
                    'history.columns': 'TRADEDATE,CLOSE'
                }
                
                try:
                    response = requests.get(base_url.format(ticker=ticker), 
                                        params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data['history']['data']:
                        chunk_df = pd.DataFrame(data['history']['data'], 
                                            columns=['date', 'close'])
                        chunk_df['date'] = pd.to_datetime(chunk_df['date'])
                        chunk_df = chunk_df.set_index('date')
                        
                        # Merge with existing data (overwrite duplicates)
                        final_df = final_df.combine_first(chunk_df)
                    
                except Exception as e:
                    failures.append((chunk_start, chunk_end, str(e)))
                
                pbar.update((chunk_end - chunk_start).days)
                chunk_start = chunk_end + timedelta(days=1)
        
        # Post-processing
        final_df.index.name = 'date'
        final_df = final_df[~final_df.index.duplicated(keep='last')]
        
        # Convert Russian decimal format and clean
        final_df['close'] = (
            final_df['close']
            .astype(str)
            .str.replace(',', '.')
            .replace('', np.nan)
            .astype(float)
        )
        
        # Forward fill missing trading days (optional)
        final_df = final_df.ffill()
        
        # # Save with metadata
        # filename = f"{ticker}_FIXED_{start_date}_to_{end_date}.xlsx"
        # with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        #     final_df.to_excel(writer, sheet_name='Prices')
            
        #     # Add error log if any
        #     if failures:
        #         pd.DataFrame(failures, columns=['Start', 'End', 'Error']).to_excel(
        #             writer, sheet_name='Errors')
        
        # print(f"\nSaved {len(final_df)} records to {filename}")
        # if failures:
        #     print(f"Warning: {len(failures)} chunks failed (see Excel 'Errors' sheet)")
        

        # deez nuts
        self.final_df =  final_df
        if _return:
            return final_df
    def _breakpoint(self, n_points:int = 100, _returns = False) -> pd.DataFrame:
        """
        Detects change points using binary segmentation and returns a pd.Series
        with 1 at breakpoints and 0 elsewhere (index matches input dates).
        
        Parameters:
            series (pd.Series): Time series with datetime index.
            n_points (int): Number of change points to detect.
        
        Returns if _return = True:
            pd.Series: Dummy series (1=breakpoint, 0=not), indexed by dates.

        """
        signal = self.final_df.values
        algo = rpt.Binseg(model="l2").fit(signal)
        bkps = algo.predict(n_points)
        
        # Initialize zeros
        dummy = pd.Series(0, index=self.final_df.index)
        # Mark breakpoints (exclude the last point, which is always the end)
        for bkp in bkps[:-1]:
            dummy.iloc[bkp] = 1
        self.dummy_brkps = dummy
        if _returns:
            return dummy
    # inflation features
    def monthly_to_daily_interpolate(self, df:pd.DataFrame)-> pd.DataFrame:
        """
        Given a DataFrame with monthly data indexed by Date,
        interpolates 'Inflation (%)' to daily values linearly,
        and forward-fills 'Inflation Target (%)' daily with the monthly value.

        Parameters:
        - df: pd.DataFrame with Date index (monthly), columns ['Inflation (%)', 'Inflation Target (%)']

        Returns:
        - daily_df: pd.DataFrame with daily index, interpolated 'Inflation (%)',
        and daily 'Inflation Target (%)' forward-filled per month.
        """
        # Ensure Date is datetime index
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Create daily date range covering full period
        daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

        # Reindex to daily frequency
        daily_df = df.reindex(daily_index)

        # Interpolate 'Inflation (%)' linearly
        daily_df['Inflation (%)'] = daily_df['Inflation (%)'].interpolate(method='linear')

        # Forward-fill 'Inflation Target (%)' per month (fill daily with monthly value)
        daily_df['Inflation Target (%)'] = daily_df['Inflation Target (%)'].ffill()

        return daily_df
    def inflation(self,_return = False, fill_type:str = 'median') -> pd.Series:
        """
        Downloads and parses inflation data and inflation target from the CBR URL.
        Returns a DataFrame with date as index and columns:
        - 'Inflation (%)'
        - 'Inflation Target (%)'
        Columns are renamed to English.
        """
        # Download the page
        url  = "https://www.cbr.ru/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2013&UniDbQuery.To=05.06.2025"
        response = requests.get(url)
        response.encoding = 'utf-8'

        # Parse all tables on the page
        tables = pd.read_html(response.text, decimal=',', thousands=' ')
        
        # Find the table with 'Инфляция' in the header
        for table in tables:
            if any('Инфляция' in str(col) for col in table.columns):
                df = table
                break
        else:
            raise ValueError("No table with 'Инфляция' found on the page.")

        # Rename columns to English
        col_map = {
            'Дата': 'Date',
            'Ключевая ставка, % годовых': 'Key Rate (%)',
            'Инфляция, % г/г': 'Inflation (%)',
            'Цель по инфляции, %': 'Inflation Target (%)'
        }
        df = df.rename(columns=col_map)

        # Prepare lists for clean data
        dates = []
        inflations = []
        inflation_targets = []

        # Iterate over rows and try to parse date and values
        for _, row in df.iterrows():
            date_str = str(row['Date'])
            inflation_str = str(row['Inflation (%)']).replace(',', '.')
            target_str = str(row['Inflation Target (%)']).replace(',', '.')
            try:
                # Parse dates in MM.YYYY format only
                date = pd.to_datetime(date_str, format='%m.%Y') + pd.offsets.MonthEnd(0)
                inflation = float(inflation_str)
                # Sometimes target might be missing or invalid, handle gracefully
                try:
                    inflation_target = float(target_str)
                except ValueError:
                    inflation_target = pd.NA
                dates.append(date)
                inflations.append(inflation)
                inflation_targets.append(inflation_target)
            except Exception:
                continue  # skip rows with invalid date or inflation

        # Build DataFrame
        result_df = pd.DataFrame({
            'Inflation (%)': inflations,
            'Inflation Target (%)': inflation_targets
        }, index=pd.DatetimeIndex(dates, name='Date'))

        result_df = result_df.sort_index()
        result_df = result_df.replace({pd.NA: np.nan})
        result_df = self.monthly_to_daily_interpolate(result_df)
        if fill_type == 'median':
            df = result_df.copy()
            column = 'Inflation Target (%)'
            df['Year'] = df.index.year
            
            # Compute yearly medians
            yearly_medians = df.groupby('Year')[column].transform('median')
            
            # Replace NaNs with yearly median
            df[column] = df[column].fillna(yearly_medians)
            
            # Drop the temporary 'Year' column
            df = df.drop(columns=['Year'])  
            self.inflation_data = df
        else:   
            self.inflation_data = result_df.dropna()
        if _return:
            return result_df.dropna()
    def inflation_gap(self, df):
        """
        Calculate the inflation gap: Inflation (%) - Inflation Target (%).
        """
        gap = df['Inflation (%)'] - df['Inflation Target (%)']
        return gap.rename('Inflation Gap (%)')
    def rolling_mean_inflation(self, df, window=6):
        """
        Calculate rolling mean of inflation over 'window' months.
        Assumes monthly data.
        """
        roll_mean = df['Inflation (%)'].rolling(window=window, min_periods=1).mean()
        return roll_mean.rename(f'Rolling Mean Inflation ({window}m)')
    def rolling_volatility_inflation(self, df, window=6):
        """
        Calculate rolling volatility (std dev) of inflation over 'window' months.
        """
        roll_vol = df['Inflation (%)'].rolling(window=window, min_periods=1).std()
        return roll_vol.rename(f'Rolling Volatility Inflation ({window}m)')
    def cumulative_inflation_year(self, df , window:int = 12):
        """
        Calculate cumulative inflation over the past 12 months.
        """
        cum_infl = df['Inflation (%)'].rolling(window=window, min_periods=1).sum()
        return cum_infl.rename('Cumulative Inflation 12m')
    def inflation_surprise(self, df):
        """
        Calculate inflation surprise: Inflation (%) - previous period's Inflation Target (%).
        """
        surprise = df['Inflation (%)'] - df['Inflation Target (%)'].shift(1)
        return surprise.rename('Inflation Surprise')
    def inflation_target_stability(self, df, window=12):
        """
        Count of inflation target changes in the past 'window' months.
        """
        changes = (df['Inflation Target (%)'].diff() != 0).astype(int)
        stability = changes.rolling(window=window, min_periods=1).sum()
        return stability.rename(f'Inflation Target Changes {window}m')
    def inflation_momentum(self,df):
        """
        Calculate month-over-month change in inflation.
        """
        momentum = df['Inflation (%)'].diff()
        return momentum.rename('Inflation Momentum')
    def inflation_target_change_flag(self,df):
        """
        Returns 1 if inflation target changed from previous period, else 0.
        """
        change_flag = (df['Inflation Target (%)'].diff().fillna(0) != 0).astype(int)
        return change_flag.rename('Inflation Target Change Flag')
    def extract_all_features(self,
                             df:pd.DataFrame,
                             rolling_mean_window=6,
                             rolling_vol_window=6,
                             cumulative_window=12,
                             stability_window=12
                             ,_return = False):
        features = pd.DataFrame(index=df.index)
        features['Inflation Gap (%)'] = self.inflation_gap(df)
        features[f'Rolling Mean Inflation ({rolling_mean_window}m)'] = self.rolling_mean_inflation(df, window=rolling_mean_window)
        features[f'Rolling Volatility Inflation ({rolling_vol_window}m)'] = self.rolling_volatility_inflation(df, window=rolling_vol_window)
        features['Inflation Target Change Flag'] = self.inflation_target_change_flag(df,)
        features[f'Cumulative Inflation {cumulative_window}m'] = self.cumulative_inflation_year(df,window=cumulative_window)
        features['Inflation Surprise'] = self.inflation_surprise(df)
        features[f'Inflation Target Changes {stability_window}m'] = self.inflation_target_stability(df,window=stability_window)
        features['Inflation Momentum'] = self.inflation_momentum(df)
        self.inflation_features = features.dropna()
        if _return:
            return self.inflation_features
    # KEY RATE FEATURES
    def changes_in_key_rate(self, _return = False) -> None:
        """
        Returns a dataframe if _returns = True:
            index = data 
            rate = percent at current data 
        ----------------------------------
        Parameters: 
            start_data 
            end_data

        """
        url = "https://www.cbr.ru/hd_base/keyrate/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2016&UniDbQuery.To=27.05.2025"
        response = requests.get(url)
        html = response.text

        # Regex pattern to match table rows with date and rate
        pattern = re.compile(r'<tr>\s*<td.*?>(\d{2}\.\d{2}\.\d{4})</td>\s*<td.*?>([\d,]+)</td>', re.DOTALL)

        matches = pattern.findall(html)

        # Prepare data for DataFrame
        data = []
        for date, rate in matches:
            rate_float = float(rate.replace(',', '.'))
            data.append({'Date': date, 'Rate': rate_float})

        # Create DataFrame
        df = pd.DataFrame(data) 

        # Optional: Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')  
        df.index = df['Date']
        df = df.drop(columns= 'Date' ,axis = 1)
        self.key_rates = df /100
        if _return:
            return self.key_rates
        
    # key rate functions 
    def rolling_max_min_rates(self,rate_series, years=1):
        """
        Compute rolling max and min rates over a window of N years.
        
        Parameters:
        - rate_series: pd.Series with datetime index
        - years: int or float, number of years for the rolling window
        
        Returns:
        - pd.DataFrame with columns 'rolling_max' and 'rolling_min'
        """
        if isinstance(rate_series, pd.DataFrame):
            rate_series = rate_series.iloc[:, 0]
        if not isinstance(rate_series, pd.Series):
            raise TypeError("rate_series must be a pandas Series")

        if rate_series.empty:
            # Return empty DataFrame with expected columns and no rows
            return pd.DataFrame(columns=['rolling_max', 'rolling_min'])
        window = int(years * 252)  # approx trading days in N years
        rolling_max = rate_series.rolling(window=window, min_periods=1).max()
        rolling_min = rate_series.rolling(window=window, min_periods=1).min()
        return pd.DataFrame({'rolling_max': rolling_max, 'rolling_min': rolling_min})

    def average_rate_change(self, rate_series, window_days=252):
        """
        Compute rolling average of absolute rate changes over a window of days.
        
        Parameters:
        - rate_series: pd.Series with datetime index
        - window_days: int, rolling window size in trading days
        
        Returns:
        - pd.Series with rolling average absolute change
        """
        abs_change = rate_series.diff().abs().fillna(0)
        avg_change = abs_change.rolling(window=window_days, min_periods=1).mean()
        cols = list(avg_change.columns)
        cols[0] = 'avg_change'
        avg_change.columns = cols
        return avg_change

    def multitude_rate_changes(self,rate_series, window_days=252):
        """
        Compute rolling count (multitude) of rate changes over a window of days.
        
        Parameters:
        - rate_series: pd.Series with datetime index
        - window_days: int, rolling window size in trading days
        
        Returns:
        - pd.Series with rolling count of changes
        """
        changes = rate_series.diff().fillna(0) != 0
        count_changes = changes.rolling(window=window_days, min_periods=1).sum()
        cols = list(count_changes.columns)
        cols[0] = 'count_changes'
        count_changes.columns = cols
        return count_changes.astype(int)
    def buildLaggedFeatures(self, s, lag=126, dropna=True):
        """
        Builds a DataFrame with lagged features for a pandas Series or DataFrame.
        Assumes ~252 trading days per year, so lag=126 corresponds to roughly 6 months.
        
        Parameters:
        - s: pd.Series or pd.DataFrame with datetime index
        - lag: number of lags to create (default 126 for ~6 months)
        - dropna: whether to drop rows with NaNs after lagging
        
        Returns:
        - DataFrame with original and lagged columns
        """
        if isinstance(s, pd.DataFrame):
            new_dict = {}
            for col_name in s.columns:
                new_dict[col_name] = s[col_name]
                for l in range(1, lag + 1):
                    new_dict[f'{col_name}_lag{l}'] = s[col_name].shift(l)
            res = pd.DataFrame(new_dict, index=s.index)
        elif isinstance(s, pd.Series):
            cols = [s.shift(i) for i in range(lag + 1)]
            res = pd.concat(cols, axis=1)
            res.columns = [f'lag_{i}' for i in range(lag + 1)]
        else:
            raise TypeError("Input must be a pandas Series or DataFrame")
        
        if dropna:
            return res.dropna()
        else:
            return res

    def rate_changed_flag(self,rate_series, window):
        """
        Creates a boolean feature indicating whether the key rate has changed
        over the last 'window' periods.
        
        Parameters:
        - rate_series: pd.Series of key rates with datetime index
        - window: integer, length of the rolling window
        
        Returns:
        - pd.Series of booleans, True if rate changed in the window, else False
        """
        # Calculate if rate changed compared to previous value
        rate_diff = rate_series.diff().fillna(0) != 0
        
        # Rolling window to check if any change occurred in the last 'window' periods
        changed_flag = rate_diff.rolling(window=window, min_periods=1).max().astype(bool)
        cols = list(changed_flag.columns)
        cols[0] = 'changed_flag'
        changed_flag.columns = cols
        return changed_flag

    def count_rate_changes(self,rate_series, window):
        """
        Counts how many times the key rate has changed in the last 'window' periods.
        
        Parameters:
        - rate_series: pd.Series of key rates with datetime index
        - window: int, rolling window size in number of periods (e.g., days)
        
        Returns:
        - pd.Series with counts of changes within the rolling window
        """
        # Identify where rate changes compared to previous value (True if changed)
        rate_changed = rate_series.diff().fillna(0) != 0
        
        # Rolling sum of changes in the window
        count_changes = rate_changed.rolling(window=window, min_periods=1).sum()
        cols = list(count_changes.columns)
        cols[0] = 'count_changes'
        count_changes.columns = cols
        return count_changes.astype(int)

    def total_rate_change_year(self,rate_series):
        """
        Calculates total absolute change in the key rate over the last 1 year (252 trading days).
        
        Parameters:
        - rate_series: pd.Series of key rates with datetime index
        
        Returns:
        - pd.Series with the rolling sum of absolute changes over 1 year
        """
        abs_change = rate_series.diff().abs().fillna(0)
        total_change = abs_change.rolling(window=252, min_periods=1).sum()
        cols = list(total_change.columns)
        cols[0] = 'total_change'
        total_change.columns = cols
        return total_change

    def average_rate_change_per_change(self,rate_series, window=252):
        """
        Calculates average rate change per change event over the last 'window' periods.
        This is a custom feature to show average magnitude of changes when they occur.
        
        Parameters:
        - rate_series: pd.Series of key rates with datetime index
        - window: int, rolling window size (default 252 ~ 1 year)
        
        Returns:
        - pd.Series with average change per change event (0 if no changes)
        """
        abs_change = rate_series.diff().abs().fillna(0)
        count_changes = self.count_rate_changes(rate_series, window)
        
        # Avoid division by zero
        avg_change = abs_change.rolling(window=window, min_periods=1).sum() / count_changes.replace(0, pd.NA)
        cols = list(avg_change.columns)
        cols[0] = 'avg_change'
        avg_change.columns = cols
        return avg_change.fillna(0)

    import pandas as pd

    def build_final_feature_dataframe(self, rate_series, lag_days=126, change_flag_window=20, count_changes_window=20, 
                                    total_change_years=1, avg_change_per_change_window=252, max_min_years=1):
        """
        Build a final DataFrame with all requested features for the key rate series.
        
        Parameters:
        - rate_series: pd.Series with datetime index of key rates
        - lag_days: int, number of lag days (~6 months default)
        - change_flag_window: int, window size for rate_changed_flag
        - count_changes_window: int, window size for count_rate_changes
        - total_change_years: int or float, years for total_rate_change_year
        - avg_change_per_change_window: int, window size for average_rate_change_per_change
        - max_min_years: int or float, years for rolling max/min
        
        Returns:
        - pd.DataFrame with all features combined
        """
        
        # 1. Lagged features
        lags_df = self.buildLaggedFeatures(rate_series, lag=256, dropna=False)
        
        # 2. Rate changed flag (boolean)
        changed_flag = self.rate_changed_flag(rate_series, window=change_flag_window)
        
        # 3. Count of rate changes in window
        count_changes = self.count_rate_changes(rate_series, window=count_changes_window)
        
        # 4. Total absolute rate change over last year(s)
        total_change = self.total_rate_change_year(rate_series)
        
        # 5. Average change per change event
        avg_change_per_change = self.average_rate_change_per_change(rate_series, window=avg_change_per_change_window)
        
        # 6. Rolling max and min rates
        max_min_df = self.rolling_max_min_rates(rate_series, years=max_min_years)
        
        # 7. Average absolute rate change
        avg_abs_change = self.average_rate_change(rate_series, window_days=int(max_min_years*252))
        
        # 8. Multitude (count) of rate changes over window
        multitude_changes = self.multitude_rate_changes(rate_series, window_days=int(max_min_years*252))
        
        # Combine all features into one DataFrame
        features = pd.concat([
            lags_df,
            changed_flag,
            count_changes,
            total_change,
            avg_change_per_change,
            max_min_df,
            avg_abs_change,
            multitude_changes
        ], axis=1)
        self.features = features
        return features


# Example usage:
# rates = pd.Series(...)  # your key rate series with datetime index
# final_features_df = build_final_feature_dataframe(rates)
# print(final_features_df.tail())

    def _pairwise_cointegration_test_filtered(self,df:pd.DataFrame = None,_return_results:bool = True) ->pd.DataFrame:
            
        """
        Tests all pairs of columns in df for cointegration using the Engle-Granger test.
        Returns only pairs cointegrated at 1%, 5%, or 10% significance.
        Returns if _return_results = True:
        - summary_df: DataFrame with only cointegrated pairs
        """
        if df is None:
            df = self.combined_df
        results = {}
        summary_rows = []
        cols = df.columns
        for col1, col2 in combinations(cols, 2):
            series1 = df[col1].dropna()
            series2 = df[col2].dropna()
            # Align indices
            joined = pd.concat([series1, series2], axis=1).dropna()
            if len(joined) < 3:
                continue  # Not enough data
            score, pvalue, _ = coint(joined.iloc[:,0], joined.iloc[:,1])
            res = {
                'pvalue': pvalue,
                'cointegrated_1%': pvalue < 0.01,
                'cointegrated_5%': pvalue < 0.05,
                'cointegrated_10%': pvalue < 0.10
            }
            # Only keep if cointegrated at any level
            if res['cointegrated_1%'] or res['cointegrated_5%']: 
            # or res['cointegrated_10%']:
                results[(col1, col2)] = res
                summary_rows.append({
                    'Pair': f"{col1} & {col2}",
                    'p-value': pvalue,
                    'Cointegrated @1%': res['cointegrated_1%'],
                    'Cointegrated @5%': res['cointegrated_5%'],
                    'Cointegrated @10%': res['cointegrated_10%']
                })
        summary_df = pd.DataFrame(summary_rows)

        self.pairwise_coint = summary_df # we save the value

        if _return_results:
            return summary_df 
    def find_cointegrated_cliques(self, df:pd.DataFrame = None, significance_level='5%' , _return: bool = False):
        # First build a graph of all cointegrated pairs
        """
        Find cliques of cointegrated series
        -----------------------------------

        """
        if df is None:
            df = self.pairwise_coint
        graph = defaultdict(set)
        col_name = f'Cointegrated @{significance_level}'
        
        for _, row in df.iterrows():
            if row[col_name]:
                a, b = row['Pair'].split(' & ')
                graph[a].add(b)
                graph[b].add(a)
        
        # Now find all maximal cliques in the graph
        def bron_kerbosch(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            for v in list(P):
                bron_kerbosch(R | {v}, P & graph[v], X & graph[v])
                P.remove(v)
                X.add(v)
        
        cliques = []
        bron_kerbosch(set(), set(graph.keys()), set())
        
        # Sort and format the results
        sorted_cliques = [sorted(clique, key=lambda x: (len(x), x)) for clique in cliques]
        
        # Remove duplicates and subsets (since we want only maximal cliques)
        unique_max_cliques = []
        for clique in sorted(sorted_cliques, key=len, reverse=True):
            clique_set = set(clique)
            if not any(clique_set.issubset(set(existing)) for existing in unique_max_cliques):
                unique_max_cliques.append(clique)
        self.unique_max_cliques = unique_max_cliques
        if _return:
            return unique_max_cliques
    def __add_series_to_df(self,col_name: str = 'close', _return:bool = False ) -> pd.DataFrame:
        """
        Adds a Series to a DataFrame as a new column aligned by index.
        
        Parameters:
        - df: pandas DataFrame with a datetime or other index
        - series: pandas Series with the same or overlapping index as df
        - col_name: string, name of the new column to add
        
        Returns:
        - df with the new column added; missing values where indices don't match
        """
        copy_for_int = self.combined_df.copy()
        # Reindex the series to match the DataFrame's index (fills missing with NaN)
        aligned_series = self.final_df.reindex(self.combined_df.index)
        
        # Assign the aligned series as a new column
        self.copy_for_int['close'] = aligned_series
        if _return:
            return self.copy_for_int
    def __filter_close_pairs(df: pd.DataFrame, pair_col: str = 'Pair') -> pd.DataFrame:
        """
        Filters rows where the 'Pair' column contains the substring 'close' (case-insensitive).
        
        Parameters:
        - df: Input DataFrame with cointegration results
        - pair_col: Name of the column containing pair strings (default: 'Pair')
        
        Returns:
        - Filtered DataFrame with only pairs containing 'close'
        """
        return df[df[pair_col].str.contains('close', case=False, na=False)]  
    def find_groups_with_keyword(self, data_dict, keyword='close'):
        """
        Iterate through each year and its groups, returning groups that contain the keyword.
        
        Parameters:
        - data_dict: dict with years as keys and list of lists as values
        - keyword: substring to search for (case-insensitive)
        
        Returns:
        - dict with years as keys and list of matching groups as values
        """
        result = {}
        for year, groups in data_dict.items():
            matching_groups = []
            for group in groups:
                # Check if any term in the group contains the keyword (case-insensitive)
                if any(keyword.lower() in term.lower() for term in group):
                    matching_groups.append(group)
            if matching_groups:
                result[year] = matching_groups
        return result  
    def add_stock_check_coint(self , ticker:str = 'SBER',iterate:bool = False) -> pd.DataFrame:
        """
        Finds whether the chosen stock series is cointegrated with interest rate
        -------------------------------------------------------------

        """
        self.moex_data(ticker) # initilizes self.final_df

        self.combine_bond_yield_csvs()

        self.__add_series_to_df() # saves self.copy_for_int
        if iterate:
            self.iterate_by_year(self.copy_for_int)
            output = self.__filter_close_pairs(self.unique_max_cliques) # we output list of lists 
            self.find_groups_with_keyword(output) # and we test it by years to find out whether it has close in it
        else:
            self._pairwise_cointegration_test_filtered(self.copy_for_that) # saves self.pairwise_coint

            self.find_cointegrated_cliques(self.pairwise_coint) # saves

            self.__filter_close_pairs(self.unique_max_cliques)

    def iterate_by_year(self, df:pd.DataFrame = None) -> list:      
        """
        Fucntion that returns cliques by year 
        -------------------------------------
        Input:
            df: self.combined
        Ouput: 
            lists of list of cliques
        """  
        if df is None:
            df = self.combined_df
        # Extract unique years from the index
        years = self.index.year.unique()
        
        for year in years:
            # Select data for the current year
            df_year = self.combined_df[self.combined_d.index.year == year]
            
            # Now you can process df_year as needed
            print(f"Year: {year}, number of rows: {len(df_year)}")
            # Example: yield or return this subset if needed
            cointegration =self.pairwise_cointegration_test_filtered(df_year)[1]
            print(self.find_cointegrated_cliques(cointegration))
    def adf_stationarity_table(self):
        """
        Performs the Augmented Dickey-Fuller test on each column of a DataFrame.
        ------------------------------------------------------------------------
        Returns a DataFrame with p-values and stationarity status at 1%, 5%, and 10% significance.
        Prints a warning if any series is stationary at the 5% level.
        """
        results = []
        stationary_found = False
        for col in self.combined_df.columns:
            series = self.combined_df[col].dropna()
            if len(series) < 3:
                continue  # Not enough data
            adf_result = adfuller(series)
            pval = adf_result[1]
            is_stationary_5 = pval < 0.05
            if is_stationary_5:
                print(f"Warning: '{col}' is stationary at the 5% significance level (p-value={pval:.4f})")
                stationary_found = True
            results.append({
                'Rate': col,
                'ADF p-value': pval,
                'Stationary @1%': pval < 0.01,
                'Stationary @5%': is_stationary_5,
                'Stationary @10%': pval < 0.10
            })
        if stationary_found:
            print("Some series are stationary at the 5% significance level.")
        return pd.DataFrame(results)
    def term_recreation(self, _automatic_return_rates: bool = False,
                           short_term_dates:Union[list, np.array] = ['3m', '6m' , '9m', '1y'], # from overall cointegration test
                           long_term_dates:Union[list, np.array] = ['10y' ,'15y' , '20y'] # from overall cointegration test
                           ) -> None:
        """
        Creates a dataframe only fro short_term values to fit a short-term solution 
        ---------------------------------------------------------------------------
        Input :
            combined_df - dataframe with all the data of rates
        Output: 
            data_short - dataframe with all the data of short term rates
            data_long- dataframe with all the data of long term rates
        """

        self.data_short = self.combined_df[short_term_dates]
        self.data_long = self.combined_df[long_term_dates]
        
    def return_short_term(self) -> pd.DataFrame:
        self.combine_bond_yield_csvs()
        self.term_recreation()
        return self.data_short
    def return_long_term(self) -> pd.DataFrame:
        self.combine_bond_yield_csvs()
        self.term_recreation()
        return self.data_long
    def key_rate_features(self,key_rate_df:pd.Series) -> pd.DataFrame:
        pass
        

