# EasyRegress

Description: A fully functional web application for performing time series forecasting, ARIMA modeling, and post-estimation diagnostics using Streamlit. The application is equipped with both manual and automatic ARIMA modeling features, and a variety of diagnostic tools to assess model assumptions and forecast accuracy. The application was designed to handle different types of time series data, including the ability to log-transform and difference the data, and visualize results interactively.

Key Features & Contributions:

  Time Series Analysis: Implemented ARIMA (AutoRegressive Integrated Moving Average) models, including manual input of ARIMA parameters and auto ARIMA for automated model selection.
  Post-Estimation Diagnostics: Integrated statistical tests such as ADF (Augmented Dickey-Fuller) test for stationarity, and various heteroscedasticity, autocorrelation, and normality tests.
  Custom Visualizations: Developed custom visualizations such as correlograms (ACF/PACF), residual plots, and forecast vs. actual charts using Matplotlib and Seaborn.
  Interactive User Interface: Created an intuitive user interface using Streamlit, allowing users to upload datasets, perform model diagnostics, and view results in real time.
  Structural Break Detection: Implemented a Chow Test to detect potential structural breaks in time series data.
  Streamlined Data Handling: Optimized data preprocessing steps (e.g., logging, differencing, lag creation) to improve model performance and accuracy. 

Python Libraries: Streamlit, Pmdarima, Statsmodels, Scikit-learn, Matplotlib, Seaborn, Pandas, Numpy
