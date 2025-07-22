# efficient_front MUST GO FIRST BEFORE capital_investment_line
# because the variables are initiated which are used by the capital_investment_line

# in the end date, the written thing must be the todays date 


import requests
import json
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import requests
import zipfile
from io import BytesIO
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.graphics.tsaplots as tsaplots 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from scipy.stats import norm
import pmdarima as pm


# Functions ----------------------------------------------------------------------------------------------------------------------------------

# Name of the page, wide layout and the picture on the web page
st.set_page_config(
    page_title = "Easy Regress App",
    page_icon = ":chart_with_upwards_trend",
    layout="wide")

# Defining a function to split the string of the stock names
@st.cache_resource
def split_string(input_string):
    # Remove commas
    cleaned_string = input_string.replace(',', '')

    # Split the cleaned string by spaces and filter out any empty strings
    separated_string = cleaned_string.split()

    # Return the list of separate words
    return separated_string

# Defining a function to load the stock data from Alpha vantage
@st.cache_resource
def load_stock_data_ind(ticker_list, interval_av, stock_type_av_index, 
                    stock_type, start_date, end_date):
    stocks_dataframe = pd.DataFrame()
    stock_dates = pd.DataFrame()
    av_api = "4XC4XAL2OFT48Y7R" # Premium
    index = 0
    for ticker in ticker_list:        
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{interval_av}&symbol={ticker}&outputsize=full&apikey={av_api}'
            r = requests.get(url) 
            data = r.json()

            # Extracting the stock data ONLY (no meta data) -------------------------------------------------------------------------------------------------------
            if interval_av == "DAILY":
                nested = data["Time Series (Daily)"]  # or whatever your top-level key is
            elif interval_av == "WEEKLY":
                nested = data["Weekly Time Series"]  # or whatever your top-level key is
            else: #interval_av == "MONTHLY":
                nested = data["Monthly Time Series"]  # or whatever your top-level key is
            
            df = pd.DataFrame.from_dict(nested, orient="index")
            df.index.name = "date"

            # Selecting only the stock type needed -----------------------------------------------------------------------------------------------------------------
            stock_data = df.iloc[:, [stock_type_av_index]] # double brackerts return a dataframe and then i can rename the columns of the series

            # Chenging the index into date object ------------------------------------------------------------------------------------------------------------------
            stock_data.index = pd.to_datetime(stock_data.index)  # convert string index to DateTimeIndex`
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Rename the column to include the stock name (e.g., "AAPL Low")
            stock_data.columns = [f"{ticker} {stock_type}"]

            first_available_date = stock_data.index[-1]
            last_available_date = stock_data.index[0]

            stock_dates.loc[index, 0] = first_available_date
            stock_dates.loc[index, 1] = last_available_date

            # Concatenating the data to the overall dataframe ------------------------------------------------------------------------------------------------------
            stocks_dataframe = pd.concat([stocks_dataframe, stock_data], axis=1)
            index += 1

            # Comparing the dates availability to the dates that the user is asking for
            if last_available_date < start_date:
                st.warning(f"{ticker} data is not available for the time period specified by the user.")
            elif start_date < last_available_date < end_date:
                st.warning(f"{ticker} available data is substantially smaller than the time period specified by the user.")
            elif start_date < first_available_date and last_available_date < end_date:
                st.warning(f"{ticker} available data is substantially smaller than the time period specified by the user.")
            elif start_date < first_available_date < end_date:
                st.warning(f"{ticker} available data is substantially smaller than the time period specified by the user.")
            elif end_date < first_available_date:
                st.warning(f"{ticker} data is not available for the time period specified by the user.")
            elif first_available_date < start_date and end_date < last_available_date:
                st.info(f"{ticker} data is available for the entire time period specified by the user")

        except Exception as e:
            st.warning(f"Could not load data for {ticker}: {e}")

    # Adding the user start and end dates to the dataframe
    stock_dates.loc[index, 0] = start_date
    stock_dates.loc[index, 1] = end_date
    stock_dates.columns = ["First available dates", "Last available dates"]

    # Adding the index names to the dataframe with the dates
    user_dates = ["USER"]
    updated_ticker_list = ticker_list + user_dates
    stock_dates.index = updated_ticker_list

    # Changing the data type into numeric
    stocks_dataframe = stocks_dataframe.apply(pd.to_numeric, errors='coerce')

    # Filtering out the datesd
    filtered_stocks_dataframe = stocks_dataframe[(stocks_dataframe.index >= start_date) & (stocks_dataframe.index <= end_date)]
    return filtered_stocks_dataframe, stock_dates, first_available_date, last_available_date, start_date, end_date


# Defining a function to calculate portfolio returns
@st.cache_resource 
def calc_port_rets(weights, stock_data):
    portfolio_returns = []
    for i in range(len(stock_data)):
        port_ret = stock_data.iloc[i, :] @ weights
        portfolio_returns.append(port_ret)
    
    portfolio_returns = pd.Series(portfolio_returns)
    portfolio_returns.index = stock_data.index

    return portfolio_returns

# Defining a function to run simple OLS
@st.cache_resource 
def simple_ols(regressand, regressors):
    #regressors = sm.add_constant(regressors)
    model = sm.OLS(regressand, regressors)
    results = model.fit()
    return results, model

# Defining a function to download Fama-French
@st.cache_resource 
def download_fama_french_factors_from_website(interval_key, start_date, end_date):
    if interval_key == 1: # Daily
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    elif interval_key == 2: # Weekly
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
    else: # Monthly
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    response = requests.get(url)
    
    # Unzipping the file
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        with z.open(z.namelist()[0]) as f:
            fama_french_data = pd.read_csv(f, skiprows=3)  # Skip metadata rows
    
    # Specify the value to check in the first column
    specific_value_1 = " Annual Factors: January-December " 
    specific_value_2 = "Copyright 2024 Kenneth R. French"

    # Find the index of the first row where the first column (A) has the specific value
    row_index_1 = fama_french_data[fama_french_data.iloc[: ,0] == specific_value_1].index
    row_index_2 = fama_french_data[fama_french_data.iloc[: ,0] == specific_value_2].index

    # If there's a row with the specific value, slice the DataFrame to remove all rows below
    if not row_index_1.empty:
        fama_french_data = fama_french_data[:row_index_1[0]] 

    # If there's a row with the specific value, slice the DataFrame to remove all rows below
    if not row_index_2.empty:
        fama_french_data = fama_french_data[:row_index_2[0]] 

    # Converting dates
    if interval_key == 1: # Daily
        fama_french_data.iloc[:, 0] = pd.to_datetime(fama_french_data.iloc[:, 0], format='%Y%m%d', errors='coerce')
    elif interval_key == 2: # Weekly
        fama_french_data.iloc[:, 0] = pd.to_datetime(fama_french_data.iloc[:, 0], format='%Y%m%d', errors='coerce')
    else: # Monthly
        fama_french_data.iloc[:, 0] = pd.to_datetime(fama_french_data.iloc[:, 0], format='%Y%m', errors='coerce')

    # Set the first column as the index
    fama_french_data = fama_french_data.set_index(fama_french_data.columns[0])

    # Changing the date format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Find the closest dates to the start_date and end_date if they don't exist
    if start_date not in fama_french_data.index:
        closest_start = fama_french_data.index.searchsorted(start_date, side="left")
        if closest_start >= len(fama_french_data.index):  # In case the start_date is beyond the data range
            closest_start = len(fama_french_data.index) - 1
        start_date = fama_french_data.index[closest_start]

    if end_date not in fama_french_data.index:
        closest_end = fama_french_data.index.searchsorted(end_date, side="right") - 1
        if closest_end < 0:  # In case the end_date is before the data range
            closest_end = 0
        end_date = fama_french_data.index[closest_end]

    # Selecting a subset between the adjusted start and end date
    fama_french_data = fama_french_data.loc[start_date:end_date]
    fama_french_data = fama_french_data.apply(pd.to_numeric, errors='coerce')

    return fama_french_data

# Defining a function to draw the efficient frontier AND the capital investment line
@st.cache_resource
def efficient_front(stocks_dataframe, rf_rate, cml = None, return_goal = None, title = None):

    # Taking the percentage change from the stock prices
    stock_data = stocks_dataframe.pct_change().copy()
    stock_data = stock_data.dropna().copy()

    # Multiply the returns by 100 to get pct points
    stock_data = stock_data * 100

    # Step for making the changes in weights
    step = 0.05
                      
    # Extracting the number of stocks
    n_of_stocks = stock_data.shape[1]

    # Get the column names (assuming stock_data has named columns)
    stock_names = stock_data.columns

    # Subsection creating all possible weight combinations 
    # Define the possible weights----------------------------------------------------------------------------------------------------------------------------
    possible_weights = [round(i * step, 2) for i in range(round(1/step) + 1)]  # From 0.0 to 1.0 in steps of 0.1

    # Generate all possible combinations of weights (three assets)
    combinations = list(itertools.product(possible_weights, repeat=n_of_stocks))

    # Filter combinations where the sum of weights equals 1
    valid_combinations = [combo for combo in combinations if sum(combo) == 1.0]

    # Convert to a Numpy for faster working for better display
    weights = np.array(valid_combinations)
    # End of the subsection -------------------------------------------------------------------------------------------------------------------------------------

    # Covariance matrix
    cov_matrix = np.cov(stock_data, rowvar=False)

    # Calculating the mean returns 
    expected_mean_returns = np.mean(stock_data, axis=0)

    # Compute standard deviations from the diagonal of the covariance matrix
    expected_stock_volatilities = np.sqrt(np.diag(cov_matrix))

    returns_array = []
    volatility_array = []
    #excess_return_array = []
    sharpe_ratio_array = []

    for i in range(len(weights)):

        # Calculate portfolio volatility: w^T * Cov * w
        port_volatility = np.sqrt((weights[i,] @ cov_matrix) @ weights[i,])

        # Appending the volatility into the list
        volatility_array.append(port_volatility)

        # Calculate portfolio return
        port_return = weights[i, :] @ expected_mean_returns

        # Appending the return into the list
        returns_array.append(port_return)
        
        # Calculating the excess return
        exc_return = port_return - rf_rate

        # Sharpe ratio
        sharpe = exc_return / port_volatility

        # Appending the sharpe 
        sharpe_ratio_array.append(sharpe)

    # Getting the extreme numbers
    largest_return = max(returns_array)
    smallest_return = min(returns_array)
    largest_volatility = max(volatility_array)
    smallest_volatility = min(volatility_array)

    # Getting the portfolio S (greatest sharpe)
    index_of_largest_sr = np.argmax(sharpe_ratio_array)
    s_port_return = returns_array[index_of_largest_sr]
    s_port_volatility = volatility_array[index_of_largest_sr]
    s_port_weights = weights[index_of_largest_sr]

    # Getting the portfolio M (lowest variance)
    index_of_lowest_risk = np.argmin(volatility_array)
    m_port_return = returns_array[index_of_lowest_risk]
    m_port_volatility = volatility_array[index_of_lowest_risk]
    m_port_weights = weights[index_of_lowest_risk]

    # Subsection for creating the a, b, f, d and alphas--------------------------------------------------------
    if cml:
        # Extracting the maximum value from the returns
        max_return = max(returns_array)

        vector_of_ones = np.ones(stock_data.shape[1])

        a = vector_of_ones @ np.linalg.inv(cov_matrix) @ expected_mean_returns
        b = expected_mean_returns @ np.linalg.inv(cov_matrix) @ expected_mean_returns
        f = vector_of_ones @ np.linalg.inv(cov_matrix) @ vector_of_ones
        d = b * f - a ** 2
        alpha_0 = (1 / d) * (b * np.linalg.inv(cov_matrix) @ vector_of_ones - a * np.linalg.inv(cov_matrix) @ expected_mean_returns)
        alpha_1 = (1 / d) * (f * np.linalg.inv(cov_matrix) @ expected_mean_returns - a * np.linalg.inv(cov_matrix) @ vector_of_ones)

        if return_goal:
            W = alpha_0 + alpha_1 * return_goal

        hypot_returns = np.linspace(0,max_return * 1.2, 251)
        hypot_weights = []
        hypot_sds = []
        for i in range(len(hypot_returns)):
            w = alpha_0 + alpha_1 * hypot_returns[i]
            hypot_weights.append(w) 
            sd = np.sqrt( w @ cov_matrix @ w )
            hypot_sds.append(sd)

        # End of the subsection ---------------------------------------------------------------------------------------------

        # Subsection for creating the capital market line  ---------------------------------------------------------------------------------------------
        # Estimating the slope
        slope = max(sharpe_ratio_array)

        # Generating values for x (standard deviation) for the straight line
        x = np.linspace(0, largest_volatility, 500)

        # Generating the y (returns) for the straight line
        y = slope * x + rf_rate

        # End of the subsection ---------------------------------------------------------------------------------------------

    # Subsection of creating a plot with feasible set----------------------------------------------------------------------------------------------------------
    # Create a scatter plot
    fig1, ax = plt.subplots(figsize = (8, 6)) 

    # Add titles and labels
    if title:
        plt.title('Feasible set')
    plt.xlabel('Volatility (Risk) [%]')
    plt.ylabel('Return [%]')

    # Colour coding
    colour_blue = (2/255, 136/255, 209/255)
    colour_orange = (255/255, 111/255, 0/255)
    colour_yellow = (254/255, 191/255, 0/255)
    colour_green = (3/255, 78/255, 65/255)

    # Create a scatter plot for FEASIBLE SET
    plt.scatter(volatility_array, returns_array, s = 5, c = colour_blue, marker = '.', label = "Possible portfolios", zorder = 2)

    if cml:
        # Create a scatter plot for EFFICIENT SET
        plt.scatter(hypot_sds, hypot_returns, s = 10, c = colour_green, marker = '.', label = "Efficient Set", zorder = 3)

        # Creating a plot for the CAPITAL MARKET LINE
        plt.plot(x, y, color=colour_blue, linewidth=1, label=f"Capital Market Line", zorder = 2)


    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate offsets as a percentage of the axis limits
    x_offsett = 0.2 * (xlim[1] - xlim[0])  # 10% of the width of the x-axis
    y_offsett = 0.2 * (ylim[1] - ylim[0])  # 10% of the height of the y-axis

   # Calculate offsets as a percentage of the axis limits
    x_offsett_label = (xlim[1] - xlim[0]) / 16
    y_offsett_label = (ylim[1] - ylim[0]) / 16 

    # Set x and y axis limits with padding 
    if cml:
        plt.xlim(-(x_offsett / 4), largest_volatility + x_offsett)  # Set the x-axis range (for volatility) 
        plt.ylim(-(y_offsett / 4), largest_return + y_offsett)  # Set the y-axis range (for returns)     
    else:
        plt.xlim(smallest_volatility - x_offsett, largest_volatility + x_offsett)  # Set the x-axis range (for volatility) 
        plt.ylim(smallest_return - y_offsett, largest_return + y_offsett)  # Set the y-axis range (for returns) 
    
    # Scatter plot for individual stocks
    plt.scatter(expected_stock_volatilities, expected_mean_returns, s = 50, c = 'red', marker = 'o', label = "Individual Stocks", edgecolors=colour_blue, zorder = 3)

    # Offsets for plotting
    #x_offset = 0.02  # Offset for horizontal position (for stock names) # this one can delete
    #y_offset = 0.005  # Offset for vertical position (for stock names) # this one can delete

    # Add labels (stock names) next to each point, positioned above the circles
    for i in range(len(stock_names)):
        plt.annotate(stock_names[i], 
        xy=(expected_stock_volatilities[i], expected_mean_returns[i]),  # Original point
        xytext=(expected_stock_volatilities[i], expected_mean_returns[i] + y_offsett_label),  # Offset for label
        fontsize=9, ha='center', va='bottom',
        bbox=dict(facecolor="red", edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))  # Arrow pointing to the point

    # Scatter plot for Portfolio S
    plt.scatter(s_port_volatility, s_port_return, s = 50, c = colour_orange, marker = 'o', label = "Portfolio S", edgecolors=colour_blue, zorder = 5)
    plt.annotate("Portfolio S", 
        xy=(s_port_volatility, s_port_return),  # Original point
        xytext=(s_port_volatility, s_port_return + y_offsett_label),  # Offset for label
        fontsize=9, ha='right', va='baseline',
        bbox=dict(facecolor=colour_orange, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='grey', lw=0.5), zorder = 5)  # Arrow pointing to the point

    # Scatter plot for Portfolio M
    plt.scatter(m_port_volatility, m_port_return, s = 50, c = colour_yellow, marker = 'o', label = "Portfolio M", edgecolors=colour_blue, zorder = 6)
    plt.annotate("Portfolio M", 
        xy=(m_port_volatility, m_port_return),  # Original point
        xytext=(m_port_volatility - x_offsett_label, m_port_return),  # Offset for label
        fontsize=9, ha='right', va='baseline',
        bbox=dict(facecolor=colour_yellow, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='grey', lw=0.5), zorder = 6)  # Arrow pointing to the point


    if cml:
        # Scatter plot for the Risk Free asset
        plt.scatter(0, rf_rate, s = 50, c = colour_orange, marker = 'o', label = "Risk Free Asset", edgecolors=colour_blue, zorder = 6)
        plt.annotate("Risk Free asset", 
            xy=(0, rf_rate),  # Original point
            xytext=(x_offsett_label * 5, y_offsett_label),  # Offset for label
            fontsize=9, ha='right', va='baseline',
            bbox=dict(facecolor=colour_orange, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), zorder = 6)  # Arrow pointing to the point
        # plt.text(m_port_volatility, m_port_return + y_offsett_label, s = "Portfolio M", fontsize = 9, ha = 'right', va = 'baseline',  
        #         color = 'black', bbox = dict(facecolor = 'orange', edgecolor = colour, boxstyle = 'round,pad = 0.3'))

    # Adding a legend
    plt.legend(loc='best', fontsize=10)

    # Show the plot
    plt.grid(True, zorder = 1)

    # Show the plot
    st.pyplot(fig1)

    n = len(weights)
    expected_mean_returns = np.array(expected_mean_returns)
    # End of the subsection ---------------------------------------------------------------------------------------------------------------------

    return (cov_matrix, s_port_volatility, s_port_return, m_port_volatility, m_port_return, expected_mean_returns, 
            expected_stock_volatilities, stock_data, m_port_weights, 
            s_port_weights, weights, n, largest_return, smallest_return, largest_volatility, smallest_volatility, 
            returns_array, volatility_array, sharpe_ratio_array)

# Defining a function to plot capital investment line (NOT USED FUNCTION)
@st.cache_resource 
def capital_investment_line(mean_returns, cov_matrix, stock_dataframe, returns_array, 
                            stock_names, rf_rate, sharpe_ratio_array, largest_volatility, 
                            s_port_volatility, s_port_return, m_port_volatility, m_port_return, 
                            expected_mean_returns, expected_stock_volatilities, volatility_array, 
                            largest_return, return_goal = None):
    
    # Subsection for creating the a, b, f, d and alphas--------------------------------------------------------
    # Extracting the maximum value from the returns
    max_return = max(returns_array)

    vector_of_ones = np.ones(stock_dataframe.shape[1])

    a = vector_of_ones @ np.linalg.inv(cov_matrix) @ mean_returns
    b = mean_returns @ np.linalg.inv(cov_matrix) @ mean_returns
    f = vector_of_ones @ np.linalg.inv(cov_matrix) @ vector_of_ones
    d = b * f - a ** 2
    alpha_0 = (1 / d) * (b * np.linalg.inv(cov_matrix) @ vector_of_ones - a * np.linalg.inv(cov_matrix) @ mean_returns)
    alpha_1 = (1 / d) * (f * np.linalg.inv(cov_matrix) @ mean_returns - a * np.linalg.inv(cov_matrix) @ vector_of_ones)

    if return_goal:
        W = alpha_0 + alpha_1 * return_goal

    hypot_returns = np.linspace(0,max_return * 1.2, 251)
    hypot_weights = []
    hypot_sds = []
    for i in range(len(hypot_returns)):
        w = alpha_0 + alpha_1 * hypot_returns[i]
        hypot_weights.append(w) 
        sd = np.sqrt( w @ cov_matrix @ w )
        hypot_sds.append(sd)

    # End of the subsection ---------------------------------------------------------------------------------------------

    # Subsection for creating the capital market line  ---------------------------------------------------------------------------------------------
    # Estimating the slope
    slope = max(sharpe_ratio_array)

    # Generating values for x (standard deviation) for the straight line
    x = np.linspace(0, largest_volatility, 500)

    # Generating the y (returns) for the straight line
    y = slope * x + rf_rate

    # End of the subsection ---------------------------------------------------------------------------------------------

    # Plotting the feasible set, efficient set and the capital market line --------------------------------------------------------
    # Colour coding
    colour_blue = (2/255, 136/255, 209/255)
    colour_orange = (255/255, 111/255, 0/255)
    colour_yellow = (254/255, 191/255, 0/255)
    colour_green = (3/255, 78/255, 65/255)

    # Create a scatter plot
    fig2, ax = plt.subplots(figsize = (8, 6)) # this is the new one

    # Plotting the main scatter plot
    plt.scatter(volatility_array, returns_array, s = 5, c = colour_blue, marker = '.', label = "Possible Portfolios", zorder = 2)

    # Plotting the efficient set
    plt.scatter(hypot_sds, hypot_returns, s = 10, c = colour_green, marker = '.', label = "Efficient Set", zorder = 3)

    # Plot the line
    #plt.scatter(x, y, s = 5, c = 'black', marker = 'x', label = f"Capital Market Line [y = {round(slope, 2)}x + {round(rf_rate, 2)}]")
    plt.plot(x, y, color=colour_blue, linewidth=1, label=f"Capital Market Line", zorder = 2)

    # Scatter plot for individual stocks
    plt.scatter(expected_stock_volatilities, expected_mean_returns, s = 50, c = 'red', marker = 'o', label = "Individual Stocks", edgecolors=colour_blue, zorder = 3)

    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate offsets as a percentage of the axis limits
    x_offsett = 0.2 * (xlim[1] - xlim[0])  # 10% of the width of the x-axis
    y_offsett = 0.2 * (ylim[1] - ylim[0])  # 10% of the height of the y-axis

   # Calculate offsets as a percentage of the axis limits
    x_offsett_label = (xlim[1] - xlim[0]) / 16
    y_offsett_label = (ylim[1] - ylim[0]) / 16 

    # Set x and y axis limits with padding
    plt.xlim(-(x_offsett / 4), largest_volatility + x_offsett)  # Set the x-axis range (for volatility) 
    plt.ylim(-(y_offsett / 4), largest_return + y_offsett)  # Set the y-axis range (for returns)   

    # Add labels (stock names) next to each point, positioned above the circles
    for i in range(len(stock_names)):
        plt.annotate(stock_names[i], 
        xy=(expected_stock_volatilities[i], expected_mean_returns[i]),  # Original point
        xytext=(expected_stock_volatilities[i], expected_mean_returns[i] + y_offsett_label),  # Offset for label
        fontsize=9, ha='center', va='bottom',
        bbox=dict(facecolor="red", edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))  # Arrow pointing to the point
        #plt.text(expected_stock_volatilities[i] + x_offset, expected_mean_returns[i] + y_offsett_label, stock_names[i], fontsize = 9, ha = 'center', va = 'bottom', 
        #         color = 'black', bbox = dict(facecolor = 'red', edgecolor = 'red', boxstyle = 'round,pad = 0.3')))

    # Scatter plot for Portfolio S
    plt.scatter(s_port_volatility, s_port_return, s = 50, c = colour_orange, marker = 'o', label = "Portfolio S", edgecolors=colour_blue, zorder = 5)
    plt.annotate("Portfolio S", 
        xy=(s_port_volatility, s_port_return),  # Original point
        xytext=(s_port_volatility, s_port_return + y_offsett_label),  # Offset for label
        fontsize=9, ha='right', va='baseline',
        bbox=dict(facecolor=colour_orange, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), zorder = 5)  # Arrow pointing to the point
    # plt.text(s_port_volatility, s_port_return + y_offsett_label, s = "Portfolio S", fontsize = 9, ha = 'right', va = 'baseline', 
    #         color = 'black', bbox = dict(facecolor = colour_orange, edgecolor = colour, boxstyle = 'round,pad = 0.3'))    

    # Scatter plot for Portfolio M
    plt.scatter(m_port_volatility, m_port_return, s = 50, c = colour_yellow, marker = 'o', label = "Portfolio M", edgecolors=colour_blue, zorder = 6)
    plt.annotate("Portfolio M", 
        xy=(m_port_volatility, m_port_return),  # Original point
        xytext=(m_port_volatility - x_offsett_label, m_port_return),  # Offset for label
        fontsize=9, ha='right', va='baseline',
        bbox=dict(facecolor=colour_yellow, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), zorder = 6)  # Arrow pointing to the point
    # plt.text(m_port_volatility, m_port_return + y_offsett_label, s = "Portfolio M", fontsize = 9, ha = 'right', va = 'baseline',  
    #         color = 'black', bbox = dict(facecolor = 'orange', edgecolor = colour, boxstyle = 'round,pad = 0.3'))

    # Scatter plot for the Risk Free asset
    plt.scatter(0, rf_rate, s = 50, c = colour_orange, marker = 'o', label = "Risk Free Asset", edgecolors=colour_blue, zorder = 6)
    plt.annotate("Risk Free asset", 
        xy=(0, rf_rate),  # Original point
        xytext=(x_offsett_label * 5, y_offsett_label),  # Offset for label
        fontsize=9, ha='right', va='baseline',
        bbox=dict(facecolor=colour_orange, edgecolor=colour_blue, boxstyle='round,pad=0.3'),
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), zorder = 6)  # Arrow pointing to the point
    # plt.text(m_port_volatility, m_port_return + y_offsett_label, s = "Portfolio M", fontsize = 9, ha = 'right', va = 'baseline',  
    #         color = 'black', bbox = dict(facecolor = 'orange', edgecolor = colour, boxstyle = 'round,pad = 0.3'))

    # Add titles and labels
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk) [%]')
    plt.ylabel('Return [%]')

    # Adding a legend
    plt.legend(loc='best', fontsize=10)

    # Show the plot
    plt.grid(True, zorder = 1)

    # Show the plot
    st.pyplot(fig2)

    # End of the subsection ------------------------------------------------------------------------------------------------------

# Defining all the functions for post estiamtion diagnostics
@st.cache_resource
def gauss_markov(fitted_values, residuals):
    st.markdown("<h2 style = 'text-align: center;'>Gauss Markov assumptions assessments</h2>", unsafe_allow_html = True)
    fig3, axes3 = plt.subplots(3, 2, figsize = (12, 4 * 3)) 

    # Adding heteroscedasticity plot
    # Plotting the residuals against the zero line
    axes3[0, 0].scatter(fitted_values, residuals, marker = '.')
    axes3[0, 0].axhline(0, color = 'red', linestyle = '--', lw = 2)
    axes3[0, 0].set_title("Scatter plot of residuals and fitted values (Het)")
    axes3[0, 0].set_xlabel("Fitted values")
    axes3[0, 0].set_ylabel("Residuals")

    # Adding serial correlation plots
    axes3[0, 1].plot(residuals, marker = '.', linestyle = 'none')
    axes3[0, 1].axhline(0, color = 'red', linestyle = '--', lw = 2)
    axes3[0, 1].set_xlabel('Observation')
    axes3[0, 1].set_ylabel('Residuals')
    axes3[0, 1].set_title('Residuals from the Regression Model (Serial Cor)')

    tsaplots.plot_acf(residuals, lags=20, ax = axes3[1, 0])
    axes3[1, 0].set_xlabel('Lags')
    axes3[1, 0].set_ylabel('Autocorrelation')
    axes3[1, 0].set_title('ACF of Residuals (Serial Cor)')

    # Step 4: Plot Partial Autocorrelation Function (PACF)
    tsaplots.plot_pacf(residuals, lags=20, ax = axes3[1, 1])
    axes3[1, 1].set_xlabel('Lags')
    axes3[1, 1].set_ylabel('Partial Autocorrelation')
    axes3[1, 1].set_title('PACF of Residuals (Serial Cor)')

    # Visualising serial correlation of errors
    # Histogram with KDE
    sns.histplot(residuals, kde=True, stat="density", linewidth=0, ax = axes3[2, 0])
    # Plot the PDF
    x_norm = np.linspace(0 - 4*1, 0 + 4*1, 1000) # where 0 is a mean and 1 is a st.dev of normal distribution
    y_norm = stats.norm.pdf(x_norm, 0, 1) # x, mean, std_dev
    axes3[2, 0].plot(x_norm, y_norm, label=f'Normal Distribution\nMean = {0}, Std Dev = {1}', color = 'black')
    axes3[2, 0].set_title("Histogram of Residuals with KDE (Norm of errs)")
    axes3[2, 0].set_xlabel('Residuals')
    axes3[2, 0].set_ylabel('Density')
    axes3[2, 0].legend()

    # Q-Q Plot
    sm.qqplot(residuals, line='45', fit=True, ax = axes3[2, 1])
    axes3[2, 1].set_title('Q-Q Plot of Residuals (Norm of errs)')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig3)

# Defining a function to create Post Estimation diagnostics
@st.cache_resource
def post_estimation_diagnostics(y_variable, x_variable, residuals, _results, name):
    # Running the diagnostics
    name = f"Post Estimation Diagnostics {name}"
    st.markdown(f"<h2 style = 'text-align: center;'>{name}</h2>", unsafe_allow_html = True)


    # Creating an empty table 
    dictionary = {
        "Heteroscedasticity test 1" : ["","","","",""],
        "Heteroscedasticity test 2" : ["","","","",""],
        "Heteroscedasticity test 3" : ["","","","",""],
        "Autocorrelation test 1" : ["","","","",""],
        "Autocorrelation test 2" : ["","","","",""],
        "Autocorrelation test 3" : ["","","","",""],
        "Autocorrelation test 4" : ["","","","",""],
        "Autocorrelation test 5" : ["","","","",""],
        "Normality test" : ["","","","",""],
        "Linearity test" : ["","","","",""]
    } 
    df = pd.DataFrame(dictionary)
    df = df.transpose()
    df.columns = ["Ho","Test", "F-Statistic", "P-Value", "Verdict"]

    # Runninng all the tests

    # Goldfeld - Quant test
    goldfeld_quandt_test = het_goldfeldquandt(y_variable, x_variable)

    # White test
    white_test = het_white(residuals, _results.model.exog)

    # Arch Test
    arch_test = het_arch(residuals)

    # Durbin Watson test
    durbin_watson_t = durbin_watson(residuals)

    # Breusch - Godfrey test
    bg_test_1 = acorr_breusch_godfrey(_results, nlags = 2)
    bg_test_2 = acorr_breusch_godfrey(_results, nlags = 3)
    bg_test_3 = acorr_breusch_godfrey(_results, nlags = 4)
    bg_test_4 = acorr_breusch_godfrey(_results, nlags = 5)

    # Jarque bera test
    jb_test = jarque_bera(residuals)

    # Ramsey reset test
    ramsey_test = linear_reset(_results, power = 3, use_f = True) # performs F-test. Can also do a T-test

    # Doing the table
    df.iloc[0,0] = "Homoscedasticity"
    df.iloc[0,1] = "Goldfeld - Quant test" 
    df.iloc[0,2] = round(goldfeld_quandt_test[0], 3)
    df.iloc[0,3] = round(goldfeld_quandt_test[1], 3) 

    df.iloc[1,0] = "Homoscedasticity"
    df.iloc[1,1] = "White test"
    df.iloc[1,2] = round(white_test[0], 3)
    df.iloc[1,3] = round(white_test[1], 3) 

    df.iloc[2,0] = "Homoscedasticity"
    df.iloc[2,1] = "Arch LM test"
    df.iloc[2,2] = round(arch_test[0], 3)
    df.iloc[2,3] = round(arch_test[1], 3)

    df.iloc[3,0] = "F.O. Autocorrelation"
    df.iloc[3,1] = "Durbin Watson test"
    df.iloc[3,2] = round(durbin_watson_t, 3)
    df.iloc[3,3] = np.nan
    df.iloc[3,4] = np.nan

    df.iloc[4,0] = "2.O. Autocorrelation"
    df.iloc[4,1] = "Breusch-Godfrey test"
    df.iloc[4,2] = round(bg_test_1[0], 3)
    df.iloc[4,3] = round(bg_test_1[1], 3)

    df.iloc[5,0] = "3.O. Autocorrelation"
    df.iloc[5,1] = "Breusch-Godfrey test"
    df.iloc[5,2] = round(bg_test_2[0], 3)
    df.iloc[5,3] = round(bg_test_2[1], 3)

    df.iloc[6,0] = "4.O. Autocorrelation"
    df.iloc[6,1] = "Breusch-Godfrey test"
    df.iloc[6,2] = round(bg_test_3[0], 3)
    df.iloc[6,3] = round(bg_test_3[1], 3)

    df.iloc[7,0] = "5.O. Autocorrelation"
    df.iloc[7,1] = "Breusch-Godfrey test"
    df.iloc[7,2] = round(bg_test_4[0], 3)
    df.iloc[7,3] = round(bg_test_4[1], 3)

    df.iloc[8,0] = "Errors are normal"
    df.iloc[8,1] = "Jarque Bera test"
    df.iloc[8,2] = round(jb_test[0], 3)
    df.iloc[8,3] = round(jb_test[1], 3)

    df.iloc[9,0] = "Func Form is Correct"
    df.iloc[9,1] = "Ramsey Reset Test"
    df.iloc[9,2] = round(ramsey_test.fvalue, 3)
    df.iloc[9,3] = round(ramsey_test.pvalue, 3)

    for i in itertools.chain(range(0, 3), range(4, 10)):
        if df.iloc[i,3] < 0.05:
            df.iloc[i,4] = "Reject Ho at 5%" 
        elif df.iloc[i,3] < 0.01:
            df.iloc[i,4] = "Reject Ho at 1%" 
        else: 
            df.iloc[i,4] = "Do not reject Ho"
    
    return df

# Function to make a table with a loss function 
@st.cache_resource
def loss_function(out_of_sample_data, forecasts, column_name, theil_number):
    mae_str = mean_absolute_error(out_of_sample_data, forecasts)
    mape_str = mean_absolute_percentage_error(out_of_sample_data, forecasts)
    rmse_str = np.sqrt(mean_squared_error(out_of_sample_data, forecasts))

    str_forecast_loss_functions = pd.DataFrame({
        "MAE": [mae_str],
        "MAPE": [mape_str],
        "RMSE": [rmse_str],
        "Theil's U": [theil_number]
        }).T
    str_forecast_loss_functions.columns = [column_name]
    return str_forecast_loss_functions

# Function to calsulate the theils u
@st.cache_resource
def theils_u(actual, forecast):

    actual = np.array(actual)
    forecast = np.array(forecast)

    # Calculate the numerators
    numerator = np.sqrt(np.mean(((forecast[1:] / actual[:-1]) - (actual[1:] / actual[:-1])) ** 2))

    # Calculate the denominator
    denominator = np.sqrt(np.mean(((actual[1:] / actual[:-1]) - 1) ** 2))

    # Theil's U calculation
    theil_u_stat = numerator / denominator

    return theil_u_stat

# Function to add lagged variables of y
@st.cache_resource
def add_lag(regressand, regressors, nol):

    # Creating a dataframe that contains everything
    df = pd.DataFrame()

    # Retrieving the name of the endogenous variable
    column_name = regressand.columns[0]
    df[column_name] = regressand
    df = pd.concat([df, regressors], axis = 1)

    # Creating a set of lagged variables
    for i in range(1, nol + 1):
        df[f"{column_name} lagged by {i}"] = df[column_name].shift(i)

    # Cleaning the NaNs
    df = df.dropna()

    # Creating a new df for x
    new_x = df.drop(columns = [column_name])

    # Creating a new df for y
    new_y = df[column_name]

    # Turning the new_y into a dataframe too
    new_y = pd.DataFrame(new_y)

    return new_y, new_x

# Defining a function for the Plotting exercise 
@st.cache_resource
def plotting_exercise(regressand, label_name, dates_checkbox = False, dates = None):
    fig3, ax = plt.subplots(figsize=(8, 6))

    if dates_checkbox:
        dates = pd.to_datetime(dates, dayfirst=True)
        last_index = len(regressand)

        # Alighning the index of the regressand and the index
        dates_copy = dates[:last_index]
        index = dates_copy

        # Plotting each column of regressand with labels
        for column in regressand.columns:
            ax.plot(index, regressand[column], label=column)
        
        #ax.plot(index, regressand) # at this moment, the only thing that is different is the name of the variable that is being plotted
        
        ax.set_title(f"Plot of {label_name}")
        ax.set_ylabel(label_name)
        #ax.set_xlabel("Dates")
        ax.set_xticks(index) 
        n_to_plot = len(index) / 10 
        n_to_plot = int(round(n_to_plot)) 
        every_other_date = index[::n_to_plot] 
        ax.set_xticks(every_other_date) 
        ax.tick_params(axis='x', rotation=45)                     
                        
    else:

        index = np.arange(len(regressand))
        
        # Plotting each column of regressand with labels
        for column in regressand.columns:
            ax.plot(index, regressand[column], label=column)
        
        #ax.plot(index, regressand)
        ax.set_title(label_name)
        ax.set_ylabel(label_name)

        #ax.set_xlabel("Observations")
        ax.set_xticks(index)
        n_to_plot = len(index) / 10
        n_to_plot = int(round(n_to_plot))
        every_other_date = index[::n_to_plot]
        ax.set_xticks(every_other_date)
        ax.tick_params(axis='x', rotation=45)  

    # Add legend to distinguish the lines
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig3)

# Defining a function for drawing histograms
@st.cache_resource
def draw_histogram(data, name, style, norm_di):

    st.markdown(f"<{style} style='text-align: center;'>{name}</{style}>", unsafe_allow_html=True)

    # Plot histogram
    fig, ax = plt.subplots()  # Create a figure and axis object for Streamlit

    # Plot histogram
    colour_blue = (2/255, 136/255, 209/255)
    counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=1, color=colour_blue, edgecolor='black', zorder=2,
                               label = f"{name}") # plotting the histogram
    
    #counts: Array of the number of data points in each bin.
    #bins: Array of bin edges.

    # Fit a normal distribution
    mean, std = np.mean(data), np.std(data)

    # Generate x values for the normal curve
    x = np.linspace(bins[0], bins[-1], 1000) # Creates 1000 evenly spaced values between the first (bins[0]) and last (bins[-1]) bin edges.
    pdf = norm.pdf(x, mean, std) # Computes the Probability Density Function (PDF) of the normal distribution with the given mean and std for each value in x.

    # Scale the normal distribution to match the histogram
    #scaled_pdf = pdf * max(counts) / max(pdf)

    # Scales the y-values of the normal distribution (pdf) to match the scale of the histogram.
    # max(counts) is the maximum frequency in the histogram.
    # max(pdf) is the peak value of the normal distribution.
    # This ensures the normal curve overlays correctly on the histogram.

    # Plot the scaled normal distribution
    #plt.plot(x, scaled_pdf, color='red', lw=2, label='Fitted Normal Distribution', zorder=2) # plotting the curve
    if norm_di:
        plt.plot(x, pdf, color='red', lw=2, 
                 label=f"Density function of a normal distribution,\nmean = {mean:.2f}, std = {std:.2f}", zorder=2) 

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()

    plt.grid(True, zorder=1)

    plt.legend(loc='upper left', fontsize=10)

    # Show the plot
    st.pyplot(fig)

# Defining a function to draw ACF and PACF
@st.cache_resource
def acf_pacf(series1, series2):
    # Plot ACF on the first subplot (axes[0])
    fig13, axes = plt.subplots(2, 1, figsize=(8, 6))
    plot_acf(series1, ax=axes[0])
    axes[0].set_title('ACF')

    # Plot PACF on the second subplot (axes[1])
    plot_pacf(series2, ax=axes[1])
    axes[1].set_title('PACF')

    # Display the figure in Streamlit
    plt.tight_layout()
    st.pyplot(fig13)

# Defining a function for the manual ARIMA model
@st.cache_resource
def conduct_manual_arima(in_sample_arima_y, one, two, three):
    model = ARIMA(in_sample_arima_y, order = (int(one), int(two), int(three)))
    return model

# Defining a function toarima_ draw the KDE with residuals plot
@st.cache_resource
def resid_kde(_model_fit):    
    residuals = _model_fit.resid[1:]
    fig, axes = plt.subplots(1,2, figsize=(8, 6))
    residuals.plot(title = "Residuals", ax = axes[0])
    residuals.plot(title = "Density", kind = "kde", ax = axes[1])
    return fig, residuals

# Defining a color coding function 
@st.cache_resource
def color_code(row):
    # Find the minimum and maximum values in the row
    min_value = row.min()
    max_value = row.max()

    # Initialize a list to store the styles
    styles = []

    # Loop through each element in the row
    for value in row:
        if value == max_value:
            styles.append('background-color: red')  # Highlight max value
        elif value == min_value:
            styles.append('background-color: green')  # Highlight min value
        else:
            styles.append('background-color: white')  # No highlight for other values

    return styles

# Defining a function to run an auto ARIMA model
@st.cache_resource
def conduct_auto_arima(in_sample_arima_y):
    auto_arima = pm.auto_arima(in_sample_arima_y, stepwise = False, seasonal = False)
    return auto_arima





# The welcome section ----------------------------------------------------------------------------------------------------------------------------------

st.write("""
         Welcome to ***Easy Regress!***

We are excited to have you here. Easy Regress is your go-to web app for ***simplifying*** portfolio analysis, data and forecasting. Whether you are working with time series data or portfolio theory or conducting in-depth regression analysis, Easy Regress is designed to make the process ***seamless and accessible***.

Some of the features include portfolio optimisation, Fama French regressions, ARIMA modeling for precise forecasting, post-estimation diagnostics for deeper insights, and customizable visualizations to help you see the ***bigger picture***. Whether you're a data expert or just getting started, Easy Regress provides the tools you need to ***analyse, predict, and succeed***.

Just open the ***data configuration centre*** on the left hand side and choose the way to upload the data! 
         
In case you choose Alpha Vantage finance, just specify the stocks tickers just like this : ***AV.L AZN.L BP.L BARC.L HSBA.L*** or ***MSFT AAPL JPM BA SHEL***. Press Enter.
        
The next step is to ***specify the dates*** that you would like to analyse. Finally, just scroll down and see what our app can do for you!
         
Our recommendation is that you choose stocks from the same country unless you can deal with the risk free rate!:)

Explore the app and let data-driven decisions become second nature!
         
         """)

# The Main Section ----------------------------------------------------------------------------------------------------------------------------------

# Section where user inputs the ticker symbols  -----------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data configuration")
    #data_options = "Alpha Vantage", "Data upload"
    data_options = "Alpha Vantage" , ""
    data_option = st.selectbox("Select the data source:", options = data_options)
    if data_option == "Data upload":
        uploaded_file = st.file_uploader("Choose a file")
    else:
        ticker_symbols = st.text_input("Write ticker symbols separated by a space only.")

# Choosing the dates  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
if ticker_symbols:

    st.markdown("<h1 style='text-align: center;'>Portfolio Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Defining a risk free rate
    rf_rate = 0.0001

    # Splitting the chosen stocks into separate stocks
    stock_names = split_string(ticker_symbols)  

    # Defining start dates
    col1, col2, col3  = st.columns(3) #start='2010-12-31'
    start_day = col1.number_input("Start day:", step = 1, min_value = 1, max_value = 31, key = "start_day")
    start_month = col2.number_input("Start month:", step = 1, min_value = 1, max_value = 12, key = "start_month")
    start_year = col3.number_input("Start year:", step = 1, min_value = 2000, max_value = 2024, key = "start_year")
    start_date = str(f"{start_year}-{start_month}-{start_day}")

    # Defining end dates
    col1, col2, col3  = st.columns(3) #end='2020-12-31'
    end_day = col1.number_input("End day:", step = 1, min_value = 1, max_value = 31, key = "end_day")
    end_month = col2.number_input("End month:", step = 1, min_value = 1, max_value = 12, key = "end_month")
    end_year = col3.number_input("End year:", step = 1, min_value = 2000, max_value = 2024, key = "end_year")
    end_date = str(f"{end_year}-{end_month}-{end_day}")

    interval_options = ["Daily", "Weekly", "Monthly"]
    interval = st.selectbox("Select your interval.", options = interval_options)
    if interval == "Daily":
        interval_av = "DAILY"
        interval_key = 1
    elif interval == "Weekly":
        interval_av = "WEEKLY"
        interval_key = 2
    else: # Monthly
        interval_av = "MONTHLY"
        interval_key = 3

    list_of_stocks = []
    list_of_options = ["Open", "High", "Low", "Close"]

    stock_type = st.selectbox("Please, choose a type of info you would like to use", options = list_of_options)
    if stock_type == "Open":
        stock_type_av = "1. open"
        stock_type_av_index = 0
    elif stock_type == "High":
        stock_type_av = "2. high"
        stock_type_av_index = 1
    elif stock_type == "Low":
        stock_type_av = "3. low"
        stock_type_av_index = 2
    else: # Close
        stock_type_av = "4. close"
        stock_type_av_index = 3

    st.markdown("---")

    # Section for loading the data for the specified stocks -------------------------------------------------------------------------------------------------------------------------------------------------------------
    portfolio_analysis = st.checkbox("Tick to start portfolio analysis.")
    if portfolio_analysis:
        stocks_dataframe, stock_dates, first_available_date, last_available_date, start_date, end_date = load_stock_data_ind(ticker_list = stock_names, interval_av = interval_av, 
                                                            stock_type_av_index = stock_type_av_index, stock_type = stock_type,
                                                            start_date = start_date, end_date = end_date)
        st.warning("Please, note that the stock data is available for the following dates.")
        #st.write(stock_dates)

        if st.checkbox("Tick to continue"):
            #st.write(stocks_dataframe.dtypes)

#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################

            # An option to show the dataframe with the chosen stocks
            with st.expander("Data preview"):
                st.dataframe(stocks_dataframe)
                st.write(f"There are {len(stocks_dataframe)} observations of data.")

            # Renaming the columns for the dataframe
            stocks_dataframe.columns = stock_names
            
            st.markdown("---")

            # Printing the efficient frontier section ------------------------------------------------------------------------------------------------------------------------------------------------------

            st.markdown("<h1 style='text-align: center;'>Efficient Frontier</h1>", unsafe_allow_html=True)
            st.info(f"The returns have been generated from the stock prices of **{', '.join(stock_names)}** for the period of **{start_date}** to **{end_date}**.")      

            # Breaking the screen into two columns with 2:1 proportion
            top_left_column, top_right_column = st.columns((2, 1))

            # Printing the efficient frontier on the top left
            with top_left_column:
                cov_matrix, s_port_volatility, s_port_return, m_port_volatility, m_port_return, expected_mean_returns, \
                expected_stock_volatilities, stock_data, m_port_weights, \
                s_port_weights, weights, n, largest_return, smallest_return, largest_volatility, smallest_volatility, \
                returns_array, volatility_array, sharpe_ratio_array = efficient_front(
                    stocks_dataframe = stocks_dataframe,  
                    rf_rate = rf_rate,
                    cml = None, 
                    return_goal = None,
                    title = True)

            # Printing the portfolio details ----------------------------------------------------------------------------------------------------------------------------------------------------------

            with top_right_column:
                st.markdown("<h3 style='text-align: center;'>Special portfolio details</h3>", unsafe_allow_html=True)
                st.write(f"There are **{len(returns_array):,}** portfolio combinations based on which the below insights are created.") 
                st.markdown("<h3 style='text-align: center;'>Special portfolio weights</h3>", unsafe_allow_html=True)

                s_and_m = pd.DataFrame()
                s_and_m["Weights for S"] = s_port_weights
                s_and_m["Weights for M"] = m_port_weights
                s_and_m["Volatilities"] = expected_stock_volatilities
                s_and_m["Mean returns"] = expected_mean_returns
                s_and_m["Stocks"] = stock_names

                # Set 'Stocks' as the index
                s_and_m = s_and_m.set_index("Stocks")

                # Use st.dataframe to make it fill the column width dynamically
                st.dataframe(s_and_m, use_container_width=True) 

                st.markdown("<h3 style='text-align: center;'>Expectations for M and S</h3>", unsafe_allow_html=True)
            
                s_and_m_rr = pd.DataFrame({
                    'Portfolio S': [s_port_return , s_port_volatility],
                    'Portfolio M': [m_port_return, m_port_volatility],
                    'Portfolios': ["Expected return", "Expected risk"],

                }, index=['Expected return', 'Expected risk'])

                s_and_m_rr = s_and_m_rr.set_index("Portfolios")

                # Display the DataFrame with the index column now as a regular column
                st.dataframe(s_and_m_rr, use_container_width=True)

            st.markdown("---")

            # Visualising the returns of portfolios S and M ----------------------------------------------------------------------------------------------------------------------------------------------------------
        
            returns_plot, returns_table = st.columns((3, 1))
            with returns_plot:
                st.markdown("<h1 style='text-align: center;'>Visuals for historic returns for portfolios S and M</h1>", unsafe_allow_html=True)
                s_portfolio_returns = calc_port_rets(weights = s_port_weights, stock_data = stock_data)
                m_portfolio_returns = calc_port_rets(weights = m_port_weights, stock_data = stock_data)

                both_portfolios = pd.DataFrame({
                    'Portfolio S': s_portfolio_returns,
                    'Portfolio M': m_portfolio_returns
                    })
                
                st.line_chart(both_portfolios)

            with returns_table:

                # Some space to visualise the S and M portfolio returns
                #with st.expander("Returns of portfolios S and M preview"):
                both_rets = pd.DataFrame({
                    "Portfolio S": s_portfolio_returns,
                    "Portfolio M": m_portfolio_returns
                })
                st.write(both_rets)

            st.markdown("---")

            # Section for creating Fama French Regressions on both portfolios --------------------------------------------------------------------------------------------------------

            # Conducting the Fama French 3 factor regression

            fama_plot, fama_table = st.columns((3,1))
            with fama_plot:
                st.markdown("<h1 style='text-align: center;'>Fama French 3 Factors time plot</h1>", unsafe_allow_html=True)

                # Loading the factors
                fama_french_factors = download_fama_french_factors_from_website(interval_key = interval_key, start_date = start_date, end_date = end_date)

                fama_french_for_review = pd.DataFrame({
                    'Market': fama_french_factors.iloc[:, 0],
                    'Size': fama_french_factors.iloc[:, 1],
                    "Value" : fama_french_factors.iloc[:, 2], 
                    "Risk Free Rate" : fama_french_factors.iloc[:, 3]           
                })
                fama_french_for_review.index.name = 'Date'
                st.line_chart(fama_french_for_review)

            with fama_table:
                # An option to review the data
                st.write(fama_french_factors)
                #st.write(f"There are {len(fama_french_factors)} observations of Fama French factors.")
                # Inform the user if the dates were adjusted

                # Adding constant to regressors
                fama_french_factors = sm.add_constant(fama_french_factors)
            st.info(f"Closest available dates selected: {start_date} to {end_date}")

            # If the lengths of portfolio returns and fama frecnh factors are not the same
            if len(s_portfolio_returns) != len(fama_french_factors):                    
                common_dates = s_portfolio_returns.index.intersection(fama_french_factors.index)
                s_portfolio_returns_aligned = s_portfolio_returns.loc[common_dates]
                m_portfolio_returns_aligned = m_portfolio_returns.loc[common_dates]
                fama_french_factors_aligned = fama_french_factors.loc[common_dates]

            # If the indices of Fama French and Portfolio returns are not aligned
            if len(s_portfolio_returns) == len(fama_french_factors): 
                if s_portfolio_returns.index != fama_french_factors.index:
                    common_dates = s_portfolio_returns.index.intersection(fama_french_factors.index)
                    s_portfolio_returns_aligned = s_portfolio_returns.loc[common_dates]
                    m_portfolio_returns_aligned = m_portfolio_returns.loc[common_dates]
                    fama_french_factors_aligned = fama_french_factors.loc[common_dates]

            st.markdown("---")
                
            # Section with visuals ---------------------------------------------------------------------------------------------------------------------------------------

            # Conducting both regressions
            s_port_results, s_port_model = simple_ols(regressand = s_portfolio_returns_aligned, regressors = fama_french_factors_aligned)
            s_port_alpha = f"{round(s_port_results.params['const'],3)}%"
            s_port_fitted_values = s_port_results.fittedvalues
            s_port_residuals = s_port_results.resid

            m_port_results, m_port_model = simple_ols(regressand = m_portfolio_returns_aligned, regressors = fama_french_factors_aligned)
            m_port_alpha = f"{round(m_port_results.params['const'],3)}%"
            m_port_fitted_values = m_port_results.fittedvalues
            m_port_residuals = m_port_results.resid

            # Breaking the page into two halves to print the regression results
            st.markdown("<h1 style='text-align: center;'>Portfolios regression results</h1>", unsafe_allow_html=True)

            left, right = st.columns(2)
            with left: 
                st.markdown("<h3 style='text-align: center;'>Portfolio S regression</h1>", unsafe_allow_html=True)
                st.write(s_port_results.summary())
            with right: 
                st.markdown("<h3 style='text-align: center;'>Portfolio M regression</h1>", unsafe_allow_html=True)
                st.write(m_port_results.summary())
            st.markdown("---")

            # Post estimation diagnostics section ----------------------------------------------------------------------------------------------------------------------------------------------------------
                    
            alpha_recommended = s_port_alpha if s_port_alpha > m_port_alpha else m_port_alpha
            portfolio_recommended = "Portfolio S" if alpha_recommended == s_port_alpha else "Portfolio M"
            options = ["Portfolio S", "Portfolio M"]
            portfolio_selected = st.selectbox("Select the portfolio you would like to analyse more:", options = options)

            # Suggestion to choose the portfolio with a greater alpha
            st.info(f"Please, note that the portfolio that generates the greater alpha is {portfolio_recommended} which generated {alpha_recommended} alpha.")

            # Section with setting the settings for the below section ----------------------------------------------------------------------------------------------------------------------------------------------------------
            if portfolio_selected == "Portfolio S":
                fitted_values_selected = s_port_fitted_values
                residuals_selected = s_port_residuals
                results_selected = s_port_results
                name = "Portfolio S"
                regressand = s_portfolio_returns_aligned
                regressand = pd.DataFrame(regressand)
                regressand.rename(columns={regressand.columns[0]: "Original returns"}, inplace=True)
                # Creating a copy for the manual ARIMA bit
                arima_regressand = regressand.copy()
                arima_regressand = pd.DataFrame(arima_regressand)
                arima_regressand.rename(columns={arima_regressand.columns[0]: "Original returns"}, inplace=True)              
                regressors = fama_french_factors_aligned

                auto_arima_regressand = arima_regressand.copy()

                # Section with the capital investment line ----------------------------------------------------------------------------------------------------------------------------------------------------------
                cml, histogram = st.columns(2)
                with cml:
                    st.markdown("<h2 style='text-align: center;'>Feasible set</h1>", unsafe_allow_html=True)
                    efficient_front(stocks_dataframe, rf_rate, cml = True, return_goal = None)
                with histogram:
                    draw_histogram(data=m_portfolio_returns_aligned, name = f"{portfolio_selected} returns", style = "h2", norm_di = True) 
        
                st.markdown("---")     

            else: 
                fitted_values_selected = m_port_fitted_values
                residuals_selected = m_port_residuals   
                results_selected = m_port_results   
                name = "Portfolio M"      
                regressand = m_portfolio_returns_aligned
                regressand = pd.DataFrame(regressand)
                regressand.rename(columns={regressand.columns[0]: "Original returns"}, inplace=True)
                # Creating a copy for the manual ARIMA bit
                arima_regressand = regressand.copy()
                arima_regressand = pd.DataFrame(arima_regressand)
                arima_regressand.rename(columns={arima_regressand.columns[0]: "Original returns"}, inplace=True)
                regressors = fama_french_factors_aligned

                # Section with the capital investment line ----------------------------------------------------------------------------------------------------------------------------------------------------------
                cml, histogram = st.columns(2)
                with cml:
                    st.markdown("<h2 style='text-align: center;'>Feasible set</h2>", unsafe_allow_html=True)
                    efficient_front(stocks_dataframe, rf_rate, cml = True, return_goal = None)
                with histogram:
                    #st.markdown("<h2 style='text-align: center;'>Feasible set</h2>", unsafe_allow_html=True)

                    draw_histogram(data=m_portfolio_returns_aligned, name = f"{portfolio_selected} returns", style = "h2", norm_di = True) 
                        
                st.markdown("---")   

            # Histograms with the fama french factors -----------------------------------------------------------------------------------------------------------------------------------
            his1, his2, his3, his4 = st.columns(4)

            with his1:
                draw_histogram(data = fama_french_factors.iloc[:, 1], name = "Market factor", style = "h2", norm_di = True)


            with his2:
                draw_histogram(data = fama_french_factors.iloc[:, 2], name = "Size factor", style = "h2", norm_di = True)

            with his3:
                draw_histogram(data = fama_french_factors.iloc[:, 3], name = "Value factor", style = "h2", norm_di = True)

            with his4:
                draw_histogram(data = fama_french_factors.iloc[:, 4], name = "Risk free factor", style = "h2", norm_di = True)

            st.markdown("---")

            # Section with actual conducting of the Gauss Markovs and the PEDs ----------------------------------------------------------------------------------------------------------------------------------------------------------
        
            gauss_markov_column, peds_column = st.columns(2)
            with gauss_markov_column: 

                GMs1 = gauss_markov(fitted_values = fitted_values_selected,
                                    residuals = residuals_selected)
            with peds_column: 

                name2 = "structural model (entire data)"
                PEDs1 = post_estimation_diagnostics(y_variable = regressand, 
                                                x_variable = regressors,
                                                residuals = residuals_selected,
                                                _results = results_selected,
                                                name = name)
                st.write(PEDs1)

            st.markdown("---")

            # Section with data transformation options ---------------------------------------------------------------------------------------------------------------------------------
            
            st.markdown("<h3 style='text-align: center;'>Data transformation options</h3>", unsafe_allow_html=True)

            nobs = len(regressand)
            index_1 = 0

            take_logs, take_differences, take_lags = st.columns(3)

            st.info(f"Please, note logs are only recommended to be chosen when the data is not stationary.")
            
    # Taking logs section -----------------------------------------------------------------------------------------------------------------------------------
            st.sidebar.header("Activity log")
            with take_logs: # Taking the logs

                # An option to take logs of both the regressand and the regressor
                str_logs = st.checkbox(f"Tick to take **logs** of {portfolio_selected}.", key = "logs_1")
                if str_logs:
                        
                    # Updating the sidebar to reflect on loaded data
                    index_1 += 1
                    st.sidebar.markdown(f"{index_1}. The **logs** of {portfolio_selected} have been taken.")

                    # Taking logs of values
                    regressand = np.log(regressand).copy()
                    regressors = np.log(regressors).copy()

                    # Replacing the infinities with None values
                    regressand.replace(-np.inf, np.nan, inplace=True) 
                    regressors.replace(-np.inf, np.nan, inplace=True) 

                    # Ensuring the first column of regressors is 1
                    regressors.iloc[:, 0] = 1

                    # Check if regressand and regressors have any None values
                    has_missing_1 = regressand.isnull().values.any() # returns true of false
                    has_missing_2 = regressors.isnull().values.any() # returns true of false

                    columns_with_missing = regressors.isnull().any() # this returned that weird table
                    columns_with_missing_names = regressors.columns[columns_with_missing] # this will get the names of the variables with None values
                    columns_with_nones = ', '.join(columns_with_missing_names) # this just contains the names of the variable        

                    # Changing all the non exisitng values which will later be deleted
                    #regressand = regressand.replace(0, np.nan)
                    #regressors = regressors.replace(0, np.nan)
                    

                    # Updating the sidebar to reflect on loaded data
                    if has_missing_1 and has_missing_2:  

                        # Taking away the NAs
                        regressand = regressand.dropna()
                        regressors = regressors.dropna()    
                        index_1 += 1
                        # Ensuring that both the regressors and the regressand have the same index
                        regressand, regressors = regressand.align(regressors, join='inner', axis=0) 
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{columns_with_nones}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)

                    elif has_missing_1:

                        # Taking away the NAs
                        regressand = regressand.dropna()
                        index_1 += 1
                        # Ensuring that both the regressors and the regressand have the same index
                        regressand, regressors = regressand.align(regressors, join='inner', axis=0) 
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{portfolio_selected}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                        
                    elif has_missing_2:

                        # Taking away the NAs
                        regressors = regressors.dropna()
                        index_1 += 1
                        # Ensuring that both the regressors and the regressand have the same index
                        regressand, regressors = regressand.align(regressors, join='inner', axis=0) 
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        st.sidebar.markdown(f"{index_1}. After taking logs, **{columns_with_nones}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)

                    # In case no Nones occur
                    else:
                        # Ensuring that both the regressors and the regressand have the same index
                        regressand, regressors = regressand.align(regressors, join='inner', axis=0)
                        # Taking away the NAs 
                        updated_nobs = min(len(regressand), len(regressors))
                        previous_nobs = nobs
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. After taking logs, **no variable** had None values.<br>The number of observations is: **{previous_nobs}**.", unsafe_allow_html=True)

                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)


    # Taking differences section ------------------------------------------------------------------------------------------------------------------------------

            with take_differences: # Taking the diffences

                # An option to take differences of regressor and the regressand 
                if str_logs:
                    take_difference = st.checkbox(f"Tick to take a **difference** of **log** {portfolio_selected}.")
                else:
                    take_difference = st.checkbox(f"Tick to take a **difference** of {portfolio_selected}.")

                if take_difference:

                    # Taking the actual differences and dropping Nones
                    regressand = regressand.diff().dropna().copy()
                    regressors = regressors.diff().dropna().copy()

                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)

                    # Retrieving the new number of observations
                    if str_logs:
                        previous_nobs = updated_nobs # the previous new becomes old
                    else:
                        previous_nobs = nobs
                    updated_nobs = len(regressand) # the new new becomes new
                    
                    # Updating the sidebar to reflect on loaded data

                    index_1 += 1

                    if str_logs:
                        st.sidebar.markdown(f"{index_1}. The difference of the **log {portfolio_selected}** has been taken.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                    else: 
                        st.sidebar.markdown(f"{index_1}. The difference of the {portfolio_selected}** has been taken.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                    
                    # Ensuring the first columns is 1 
                    regressors.iloc[:, 0] = 1

    # Section with adding lags -----------------------------------------------------------------------------------------------------------------------------------------

            with take_lags:
                
                nol = st.number_input("Enter the number of lagged variables you need.", 
                                        min_value = 0,
                                        step = 1)
                if nol:
                    regressand, regressors = add_lag(regressand = regressand,
                                                    regressors = regressors,
                                                    nol = nol)
                    
                    # Resetting the index of the regressand and the regressors
                    regressand = regressand.reset_index(drop=True)
                    regressors = regressors.reset_index(drop=True)
                    
                    # Retrieving the new number of observations
                    if str_logs and take_difference:
                        previous_nobs = updated_nobs 
                    elif str_logs:
                        previous_nobs = updated_nobs
                    elif take_difference:
                        previous_nobs = updated_nobs
                    else:
                        previous_nobs = nobs
                    updated_nobs = len(regressand)

                    # Updating the sidebar to reflect on loaded data
                    index_1 += 1
                    if str_logs:  
                        st.sidebar.markdown(f"{index_1}. The number of lagged variables of {portfolio_selected} is **{nol}**.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                    else:
                        st.sidebar.markdown(f"{index_1}. The number of lagged variables of {portfolio_selected} is **{nol}**.<br>The new number of observations is: **{updated_nobs} (old: {previous_nobs})**.", unsafe_allow_html=True)
                else:
                    nol = 0

            st.markdown("---")

            # Section creating out of sample forecasts ----------------------------------------------------------------------------------------------------------------------------------------------------------
            regressors = sm.add_constant(regressors)
            
            in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7)

            if in_sample_proportion:
                if not "updated_nobs" in globals():
                    updated_nobs = nobs

                out_of_sample_proportion = 1 - in_sample_proportion
                in_sample_nobs = round(updated_nobs * in_sample_proportion) 

                # Updating the sidebar to reflect on loaded data
                st.info(f"The in sample proportion is selected at **{round(in_sample_proportion * 100)}%**. The in sample number of observations is: **{in_sample_nobs} (before data splitting: {updated_nobs})**. The number of observations to forecast is: **{updated_nobs - in_sample_nobs}**.") 

                # Now will break all data into in sample and out of sample data
                # Creating in sample Y and out of sample Y
                in_sample_y, out_of_sample_y = regressand.iloc[:in_sample_nobs].copy(), regressand.iloc[in_sample_nobs:].copy()

                # Creating in sample X and out of sample X
                in_sample_x, out_of_sample_x = regressors.iloc[:in_sample_nobs, :].copy(), regressors.iloc[in_sample_nobs:, :].copy()
    
                # Section with making an in sample regression ----------------------------------------------------------------------------------------------------------------------------------------------------------
                fitted_model, model = simple_ols(regressand = in_sample_y, regressors = in_sample_x)  

                st.markdown("<h1 style='text-align: center;'>In sample and out of sample analysis</h1>", unsafe_allow_html=True)
                str_forecast_graph, in_sample_regression  = st.columns(2)


                with str_forecast_graph:
                    # FORECASTING the next nobs_to_predict Y variable
            
                    str_forecasts = fitted_model.predict(out_of_sample_x)

                    # Comparing the REAL NUMBERS with PREDICTIONS section -------------------------------------------------------------------------------------------------------------------------------------------------
                    if regressors.shape[1] > 1:

                        # Plotting the actual Y together with the forecast
                        #st.markdown("<h1 style='text-align: center;'>Comparing actual data with out of sample forecasts</h1>", unsafe_allow_html=True)     

                        regressand_2 = regressand.copy()
                        regressand_2[f"{portfolio_selected} returns forecast"] = [None]*len(in_sample_y) + list(str_forecasts)
                        fig1, ax = plt.subplots()

                        # Plotting exercise
                        if str_logs and take_difference:   
                            label_name_1 = f"Differenced log {portfolio_selected}"
                        elif str_logs:
                            label_name_1 = f"Log {portfolio_selected}"
                        elif take_difference:
                            label_name_1 = f"Differenced {portfolio_selected}"
                        else:
                            label_name_1 = f"{portfolio_selected}" 

                        regressand_2.rename(columns={regressand.columns[0]: label_name_1}, inplace=True)
                        
                        plotting_exercise(regressand = regressand_2,
                                            label_name = label_name_1)
                        
                        # Conducting the loss functions

                        name_1 = "Structural model"
                        theils_1 = theils_u(actual = out_of_sample_y, forecast = str_forecasts)
                        loss_functions_1 = loss_function(out_of_sample_data = out_of_sample_y, forecasts = str_forecasts, column_name = name_1, theil_number = theils_1)
                        #st.markdown("<h3 style='text-align: center;'>Loss functions from structural model</h1>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.dataframe(loss_functions_1, use_container_width=True)

                with in_sample_regression:
                    st.write(fitted_model.summary())

                st.markdown("---")





























                # Manual ARIMA section --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                manual_arima = st.checkbox("Please, tick if you would like to estimate a manual ARIMA model.")
                if manual_arima:
                    arima_previous_nobs = len(arima_regressand)
                    st.markdown("---")
                    # Updating the sidebar to reflect on loaded data t
                    st.sidebar.markdown("<h1 style='text-align: center;'>ARIMA  Manual Analysis</h1>", unsafe_allow_html=True)
                    st.markdown("<h1 style='text-align: center;'>ARIMA Analysis</h1>", unsafe_allow_html=True)

            # Jox Jenkings part one section ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    st.markdown("---")
                    st.markdown("<h1 style='text-align: center;'>Box Jenkins Methodology</h1>", unsafe_allow_html=True)

            # Section with taking logs ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    arima_logs = st.checkbox(f"Tick to take **logs** of {portfolio_selected} returns.", key = "logs_2")
                    if arima_logs:

                        # Updating the sidebar to reflect on loaded data
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The **logs** of {portfolio_selected} returns have been taken.")

                        arima_regressand = np.log(arima_regressand).copy()

                        # Ensuring the None values are gone
                        has_missing_3 = arima_regressand.isnull().values.any() # returns true of false
                        if has_missing_3:
                            arima_regressand = arima_regressand.dropna()

                            # Resetting the index of the arima_regressand and getting the last index of the arima_regressand
                            arima_regressand = arima_regressand.reset_index(drop=True)
                            
                            # Extracting the new number of observations IF there are Nones
                            arima_updated_nobs = len(arima_regressand)

                            # Making a comment in the sidebar
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. After taking logs, **{portfolio_selected}** returns had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**.", unsafe_allow_html=True)
                        else:
                            # Making a comment in the sidebar
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. After taking logs, **{portfolio_selected}** returns had no None values.<br>The number of observations is still: **{arima_previous_nobs})**.", unsafe_allow_html=True)

                            # Extracting the new number of observations IF there are Nones
                            arima_updated_nobs = len(arima_regressand)

            # First graph, correlograms and ADF section ------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    graph, correlograms, adf_test_3 = st.columns((2,2,1))
                    with graph: 
                        if arima_logs:
                            label_name_8 = f"Log {portfolio_selected} returns"
                        else: 
                            label_name_8 = f"{portfolio_selected} returns"

                        plotting_exercise(regressand = arima_regressand,
                                            label_name = label_name_8)
                    with correlograms:
                        acf_pacf(series1 = arima_regressand, series2 = arima_regressand)

                    with adf_test_3:
                        st.markdown("<h2 style='text-align: center;'>Augmented Dickey Fuller test</h2>", unsafe_allow_html=True)

                        adf_test_1 = adfuller(arima_regressand)

                        if adf_test_1[1] > 0.05:
                            st.write("")
                            st.write(f"p value is **{round(adf_test_1[1], 3)}**.")
                            st.write(f"Hence, There is not enough information to reject Ho. \n**We have evidence of a unit root**.")
                            st.write("**Data is not stationary**")
                        else:
                            st.write("")
                            st.write(f"p value is **{round(adf_test_1[1], 3)}**. Hence, we have enough evidence to reject Ho. \nThere is **no evidence of a unit root**.")
                            st.write(f"**Data is stationary**.")

                    st.markdown("---")

            # Breaking the data into in sample and out of sample data section ------------------------------------------------------------------------------------------

                    arima_in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7, key = "arima split 2")
                    
                    if arima_in_sample_proportion:
                        if not "arima_updated_nobs" in globals():
                            arima_updated_nobs = nobs
                        
                        out_of_sample_proportion_2 = 1 - arima_in_sample_proportion
                        arima_in_sample_nobs = round(arima_updated_nobs * arima_in_sample_proportion) 

                        # Updating the sidebar to reflect on loaded data
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The in sample proportion is selected at **{round(arima_in_sample_proportion * 100)}%**. Out of sample proportion is selected at **{round(100 - arima_in_sample_proportion * 100)}%**.", unsafe_allow_html=True) 
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The in sample number of observations is: **{arima_in_sample_nobs} (before data splitting: {arima_updated_nobs})**. The number of observations to forecast is: **{arima_updated_nobs - arima_in_sample_nobs}**.", unsafe_allow_html=True) 

                        # Splitting the data into in sample and out of sample data
                        in_sample_arima_y, out_of_sample_arima_y = arima_regressand.iloc[:arima_in_sample_nobs].copy(), arima_regressand.iloc[arima_in_sample_nobs:].copy()

                        # Checking for stationarity
                        st.markdown("---")
                        if arima_logs:
                            st.markdown(f"<h1 style = 'text-align: center;'>Checking log {portfolio_selected} returns for stationarity (ADF test)</h1>", unsafe_allow_html = True)    
                        else:
                            st.markdown(f"<h1 style = 'text-align: center;'>Checking {portfolio_selected} returns for stationarity (ADF test)</h1>", unsafe_allow_html = True)

            # Second graph, correlograms and ADF section ------------------------------------------------------------------------------------------------------------------------------------------------------------------

                        # Plotting the first round of correlograms
                        if arima_logs:
                            st.markdown(f"<h1 style = 'text-align: center;'>Visuals and correlograms for log {portfolio_selected} returns</h1>", unsafe_allow_html = True)    
                        else:
                            st.markdown(f"<h1 style = 'text-align: center;'>Visuals and correlograms for {portfolio_selected} returns</h1>", unsafe_allow_html = True)

            # Correlograms after splitting the data section -------------------------------------------------------------------------------------------------------------
                        
                        graph2, correlograms2, adf_test_4= st.columns((2,2,1))
                        with graph2:
                            plotting_exercise(regressand = in_sample_arima_y,
                                    label_name = label_name_8)
                    
                        with correlograms2:
                            # Plotting ACF and PACF
                            acf_pacf(series1 = in_sample_arima_y, series2 = in_sample_arima_y)

                        with adf_test_4:

                            st.markdown("<h2 style='text-align: center;'>Augmented Dickey Fuller test</h2>", unsafe_allow_html=True)

                            # Conducting the first round of Augmented Dickey Fuller test
                            adf_test_6 = adfuller(in_sample_arima_y)

                            if adf_test_6[1] > 0.05:
                                st.write("")
                                st.write(f"p value is **{round(adf_test_6[1], 3)}**.")
                                st.write(f"Hence, There is not enough information to reject Ho. \n**We have evidence of a unit root**.")
                                st.write("**Data is not stationary**")
                            else:
                                st.write("")
                                st.write(f"p value is **{round(adf_test_6[1], 3)}**. Hence, we have enough evidence to reject Ho. \nThere is **no evidence of a unit root**.")
                                st.write(f"**Data is stationary**.")
                        st.markdown("---")

            # Taking differences and plotting the correlograms again section ----------------------------------------------------------------------------------------------------

                        # Updating the sidebar to reflect on loaded data
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. Please, note that the differences are taken **solely** for visualisation purposes. The data will **not** be differenced in the stage of manual ARIMA estimation", unsafe_allow_html = True)   

                        # Taking the difference
                        in_sample_arima_y_diff = in_sample_arima_y.diff().dropna().copy()

                        # Resetting the index of in_sample_arima_y_diff
                        in_sample_arima_y_diff = in_sample_arima_y_diff.reset_index(drop=True)

                        # Extracting the new number of observations
                        if arima_logs:
                            arima_previous_nobs = arima_updated_nobs
                        arima_updated_nobs = len(in_sample_arima_y_diff)
                        
                        if arima_logs:
                            st.markdown(f"<h1 style = 'text-align: center;'>Visuals and correlograms of differenced log {portfolio_selected} returns</h1>", unsafe_allow_html = True)
                            # Updating the sidebar to reflect on loaded data
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. The **difference** of the **log {portfolio_selected} returns** has been taken. <br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**", unsafe_allow_html = True)                
                        
                        else:
                            st.markdown(f"<h1 style = 'text-align: center;'>Visuals and correlograms of differenced {portfolio_selected} returns</h1>", unsafe_allow_html = True)
                            # Updating the sidebar to reflect on loaded data
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. The **difference** of the **{portfolio_selected} returns** has been taken. <br>The new number of observations is: **{arima_updated_nobs} (old: {arima_previous_nobs})**", unsafe_allow_html = True)                
                        
                        # Plotting exercise
                        graph2, correlograms2, adf_test_7 = st.columns((2,2,1))
                        with graph2:

                        # Plotting exercise
                            if arima_logs:   
                                label_name_2 = f"Differenced log {portfolio_selected} returns" # diffrencesof the log data
                            else:
                                label_name_2 = f"Differenced {portfolio_selected} returns" # diffrencesof the log data

                            in_sample_arima_y_diff_copy = in_sample_arima_y_diff.copy()
                            in_sample_arima_y_diff_copy.rename(columns={in_sample_arima_y_diff_copy.columns[0]: f"Differenced {portfolio_selected} returns"}, inplace=True)
                            plotting_exercise(regressand = in_sample_arima_y_diff_copy,
                                        label_name = label_name_2)
                        
                        with correlograms2:

                            # Plotting ACF and PACF
                            acf_pacf(series1 = in_sample_arima_y_diff, series2 =in_sample_arima_y_diff) 
                        
                        with adf_test_7:
                            st.markdown("<h2 style = 'text-align: center;'>Dickey fuller test on differenced data</h2>", unsafe_allow_html = True)
                            adf_test_2 = adfuller(in_sample_arima_y_diff)

                            if adf_test_2[1] > 0.05:
                                st.write("")
                                st.write(f"p value is **{round(adf_test_2[1], 3)}**.")
                                st.write(f"Hence, There is not enough information to reject Ho. \n**We have evidence of a unit root**.")
                                st.write("**Data is not stationary**")
                            else:
                                st.write("")
                                st.write(f"p value is **{round(adf_test_2[1], 3)}**. Hence, we have enough evidence to reject Ho. \nThere is **no evidence of a unit root**.")
                                st.write(f"**Data is stationary**.")
                        
                        st.markdown("---")

            # The regression section ---------------------------------------------------------------------------------------------------------------------------
                        # Conducting a regression for the manual model

                        st.markdown("<h1 style='text-align: center;'>Step 2: Estimation</h1>", unsafe_allow_html=True)

                        # Breaking the screen into three columnds and asking to submit three numbers
                        col1, col2, col3  = st.columns(3)
                        one = col1.number_input("First num?:", step = 1, min_value = 0)
                        two = col2.number_input("Second num?:", step = 1, min_value = 0)
                        three = col3.number_input("Third num?:", step = 1, min_value = 0)

                        # Fitting an ARIMA model
                        if one == 0 and two == 0 and three == 0:
                            st.write("Please, indicate your numbers for ARIMA guess.")
                            model_fit = None
                            
                        elif (one is not None) and (two is not None) and (three is not None):
                            model = conduct_manual_arima(in_sample_arima_y = in_sample_arima_y,
                                                        one = one, 
                                                        two = two, 
                                                        three = three)

                            model_fit = model.fit()
                                # st.write(model_fit.summary())   
                            st.markdown("---")  

                            # Updating the sidebar to reflect on loaded data
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. **Manual** ARIMA model of order **{one, two, three}** has been estimated.")                

                            # Retrieving fitted values and residuals
                            fitted_values_manual_arima = model_fit.fittedvalues
                            residuals_manual_arima = model_fit.resid

            # Plotting the regression results and the forecast plot section -----------------------------------------------------------------------------------------------
                            manual_arima_regression, manual_arima_forecast = st.columns(2)
                            with manual_arima_regression:
                                st.write(model_fit.summary())   

                            with manual_arima_forecast:
                                # Forecasting and plotting the forecast from the manual procedure
                                arima_manual_forecasts = model_fit.forecast(len(out_of_sample_arima_y))

                                arima_regressand["ARIMA manual forecasts"] = [None]*len(in_sample_arima_y) + list(arima_manual_forecasts)
                                
                                # Plotting exercise
                                if arima_logs:   
                                    label_name_3 = f"Log {portfolio_selected} returns"
                                else:
                                    label_name_3 = f"{portfolio_selected} returns"

                                # Plotting exercise
                                plotting_exercise(regressand = arima_regressand,
                                            label_name = label_name_3)
                            st.markdown("---")

                        else:
                            st.write("Please, indicate your numbers for ARIMA guess")
                            st.markdown("---")

            # Plotting residuals and their density section ----------------------------------------------------------------------------------------------------------------
                        # Plotting the residuals and their density function
                        if model_fit:
                            st.markdown("<h1 style = 'text-align: center;'>Residual density function and residuals correlogram</h1>", unsafe_allow_html = True)
                            residuals_and_density, residuals_correlograms = st.columns(2)
                            with residuals_and_density:
                                #st.markdown("<h1 style = 'text-align: center;'>Plotting the residuals and density</h1>", unsafe_allow_html = True)
                                fig4, residuals = resid_kde(_model_fit = model_fit)
                                plt.tight_layout()
                                st.pyplot(fig4)

                            with residuals_correlograms:
                                # Checking residuals for the presence of autocorrelation
                                # Errors (residuals) must be a white noise ie must not have any autocorrelation
                                #st.markdown("<h1 style = 'text-align: center;'>Correlograms for residuals</h1>", unsafe_allow_html = True)

                                # Plot ACF and PACF
                                acf_pacf(series1 = residuals_manual_arima, series2 = residuals_manual_arima)
                            st.markdown("---")

            # Calculating the loss functions for manual ARIMA model section -------------------------------------------------------------------------------------------------
                            
                            #st.markdown("<h1 style = 'text-align: center;'>Statistical loss functions for manual ARIMA model</h1>", unsafe_allow_html = True)
                            name2 = "Manual ARIMA model"
                            theils_2 = theils_u(actual = out_of_sample_arima_y, forecast = arima_manual_forecasts)
                            loss_functions_2 = loss_function(out_of_sample_data = out_of_sample_arima_y, forecasts = arima_manual_forecasts, column_name = name2, theil_number = theils_2)
                            #st.write(loss_functions_2)

                            # Comparing all existing loss functions for all created forecasting models                         

                            list_of_loss_functions = []
                            if "loss_functions_1" in globals():
                                list_of_loss_functions.append(loss_functions_1)
                            if "loss_functions_2" in globals():
                                list_of_loss_functions.append(loss_functions_2)
                            #if "loss_functions_3" in globals():
                                #list_of_loss_functions.append(loss_functions_3)
                            df = pd.DataFrame()

                            if len(list_of_loss_functions) > 1:
                                #if st.checkbox("Tick to compare the loss functions of all created forecasting models.", key = "1"):
                                st.markdown("<h1 style = 'text-align: center;'>Comparing existing loss functions</h1>", unsafe_allow_html = True)
                    
                                for i in list_of_loss_functions:
                                    df = pd.concat([df, i], axis = 1)
                                styled_df = df.style.apply(color_code, axis=1)                                  
                                st.markdown(styled_df.to_html(), unsafe_allow_html=True)
                st.markdown("---")     


                # Automatic ARIMA section --------------------------------------------------------------------------------------------------------------------------------------------

                do_automatic_arima = st.checkbox("Tick to create an automatic ARIMA model.")
                st.info(f"**Please, note that this make take several minutes to complete.**")
                
                if do_automatic_arima:

                    # Making a comment in the sidebar
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("<h1 style='text-align: center;'>ARIMA  Automatic Analysis</h1>", unsafe_allow_html=True)
                
                    # Retrieving the number of observations
                    auto_arima_previous_nobs = len(auto_arima_regressand)                 

                    # An option to take logs

                    auto_arima_logs = st.checkbox(f"Tick to take **logs** of {portfolio_selected}.", key = "logs_3")
                    if auto_arima_logs:
                        
                        # Taking the actual logs
                        auto_arima_regressand = np.log(auto_arima_regressand).copy()

                        # Updating the sidebar to reflect on loaded data
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The **logs** of {portfolio_selected} have been taken.")

                        # Ensuring the None values are gone
                        has_missing_4 = auto_arima_regressand.isnull().values.any() # returns true of false
                        if has_missing_4:
                            auto_arima_regressand = auto_arima_regressand.dropna()

                            # Resetting the index of the arima_regressand and getting the last index of the arima_regressand
                            auto_arima_regressand = auto_arima_regressand.reset_index(drop=True)
                            
                            # Extracting the new number of observations IF there are Nones
                            auto_arima_updated_nobs = len(auto_arima_regressand)

                            # Making a comment in the sidebar
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. After taking logs, **{portfolio_selected}** had None values. They were excluded for the regression purposes.<br>The new number of observations is: **{auto_arima_updated_nobs} (old: {auto_arima_previous_nobs})**.", unsafe_allow_html=True)
                        else:
                            # Making a comment in the sidebar
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. After taking logs, **{portfolio_selected}** had no None values.<br>The number of observations is still: **{auto_arima_previous_nobs})**.", unsafe_allow_html=True)

                            # Extracting the new number of observations IF there are Nones
                            auto_arima_updated_nobs = len(auto_arima_regressand)

                        st.markdown(f"<h1 style = 'text-align: center;'>Log {portfolio_selected}</h1>", unsafe_allow_html = True)    
                        
                        # Plotting exercise
                        label_name_6 = f"Log {portfolio_selected}"
                        plotting_exercise(regressand = auto_arima_regressand,
                                    label_name = label_name_6)     

                    #st.markdown("---")
            
                    # No need to take differences because, the automatic ARIMA will choose this degree itself
                    # Breaking the data into in sample and out of sample periods

                    auto_arima_in_sample_proportion = st.number_input("Proportion for in-sample data", step = 0.05, value = 0.7, key = "auto arima split 2")
                
                    if auto_arima_in_sample_proportion:
                        if not "auto_arima_updated_nobs" in globals():
                            auto_arima_updated_nobs = auto_arima_previous_nobs
                        
                        out_of_sample_proportion_3 = 1 - auto_arima_in_sample_proportion
                        auto_arima_in_sample_nobs = round(auto_arima_updated_nobs * auto_arima_in_sample_proportion) 

                        # Updating the sidebar to reflect on loaded data
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The in sample proportion is selected at **{round(auto_arima_in_sample_proportion * 100)}%**. Out of sample proportion is selected at **{round(100 - auto_arima_in_sample_proportion * 100)}%**.", unsafe_allow_html=True) 
                        index_1 += 1
                        st.sidebar.markdown(f"{index_1}. The in sample number of observations is: **{auto_arima_in_sample_nobs} (before data splitting: {auto_arima_updated_nobs})**. The number of observations to forecast is: **{auto_arima_updated_nobs - auto_arima_in_sample_nobs}**.", unsafe_allow_html=True) 

                        # Splitting the data into in sample and out of sample data
                        in_sample_auto_arima_y, out_of_sample_auto_arima_y = auto_arima_regressand.iloc[:auto_arima_in_sample_nobs].copy(), auto_arima_regressand.iloc[auto_arima_in_sample_nobs:].copy()


                        st.markdown("<h1 style = 'text-align: center;'>Automatic ARIMA regression results</h1>", unsafe_allow_html = True)    
                        auto_arima = conduct_auto_arima(in_sample_auto_arima_y)

                        st.write(f"Automatic ARIMA model suggests that the model is: {auto_arima}")

                        auto_arima_regr, auto_arima_graph = st.columns(2)
                        with auto_arima_regr:
                            st.write(auto_arima.summary())

                        with auto_arima_graph:
                            # Updating the activity log to say about the auto arima
                            index_1 += 1
                            st.sidebar.markdown(f"{index_1}. Automatic ARIMA model suggests that the model is: **{auto_arima.order}**.")

                            # Generating and plotting the results
                            #st.markdown("<h1 style = 'text-align: center;'>Plot of automatic forecasts</h1>", unsafe_allow_html = True)    
                            arima_auto_forecasts = auto_arima.predict(n_periods = len(out_of_sample_auto_arima_y))
                            auto_arima_regressand["arima_auto_forecasts"] = [None] * len(in_sample_auto_arima_y) + list(arima_auto_forecasts)
                    
                            # Plotting exercise
                            if auto_arima_logs:   
                                label_name_7 = f"Log {portfolio_selected}"
                            else:
                                label_name_7 = f"{portfolio_selected}"

                            # Plotting exercise
                            plotting_exercise(regressand = auto_arima_regressand,
                                                label_name = label_name_7)

                    st.markdown("---")

                    # Calculating the loss functions for manual ARIMA model                         
                    #st.markdown("<h1 style = 'text-align: center;'>Statistical loss functions for manual ARIMA model</h1>", unsafe_allow_html = True)
                    name2 = "Automatic ARIMA model"
                    theils_2 = theils_u(actual = out_of_sample_auto_arima_y, forecast = arima_auto_forecasts)
                    loss_functions_3 = loss_function(out_of_sample_data = out_of_sample_auto_arima_y, 
                                                        forecasts = arima_auto_forecasts, 
                                                        column_name = name2, 
                                                        theil_number = theils_2)
                    #st.write(loss_functions_3)
                    #st.markdown("---")

                    # Comparing all existing loss functions for all created forecasting models 

                    list_of_loss_functions = []
                    if "loss_functions_1" in globals():
                        list_of_loss_functions.append(loss_functions_1)
                    if "loss_functions_2" in globals():
                        list_of_loss_functions.append(loss_functions_2)
                    if "loss_functions_3" in globals():
                        list_of_loss_functions.append(loss_functions_3)
                    df = pd.DataFrame()

                    if len(list_of_loss_functions) > 1:
                        st.markdown("<h1 style = 'text-align: center;'>Comparing existing loss functions</h1>", unsafe_allow_html = True)
            
                        for i in list_of_loss_functions:
                            df = pd.concat([df, i], axis = 1)
                        styled_df = df.style.apply(color_code, axis=1)                                  
                        st.markdown(styled_df.to_html(), unsafe_allow_html=True)
                        st.markdown("---")