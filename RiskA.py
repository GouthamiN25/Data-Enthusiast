import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import datetime

# Download historical data for a set of stocks
tickers = ['AAPL', 'GOOG', 'AMZN']
start_date = '2023-01-01'
end_date = '2023-12-31'

data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

returns.head()

# Basic Exploratory Data Analysis
plt.figure(figsize=(14, 7))
sns.lineplot(data=returns)
plt.title('Daily Returns of Stocks')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend(tickers)
plt.show()

# Calculate risk metrics
mean_returns = returns.mean()
volatility = returns.std()
print(f'Mean Returns:\n{mean_returns}\n')
print(f'Volatility (Standard Deviation):\n{volatility}\n')

# Calculate Value at Risk (VaR) at 95% confidence level
VaR_95 = returns.quantile(0.05)
print(f'Value at Risk (VaR) at 95% confidence level:\n{VaR_95}\n')

# Calculate Conditional Value at Risk (CVaR) at 95% confidence level
CVaR_95 = returns[returns < VaR_95].mean()
print(f'Conditional Value at Risk (CVaR) at 95% confidence level:\n{CVaR_95}\n')

# Fit ARIMA model for volatility prediction
stock = 'AAPL'
returns_stock = returns[stock]

model = ARIMA(returns_stock, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Forecast volatility
forecast_volatility = model_fit.forecast(steps=30)
plt.figure(figsize=(10, 5))
plt.plot(forecast_volatility, label='Forecasted Volatility')
plt.title('Forecasted Volatility for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Forecasted Volatility')
plt.legend()
plt.show()

# Prepare data for regression model
returns_shifted = returns.shift(-1).dropna()
X = returns[:-1]
y = returns_shifted['AAPL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Plotting predictions vs actual using a bar plot
plt.figure(figsize=(14, 7))

# Create a DataFrame to hold the actual and predicted returns
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

# Plot the bar plot
comparison_df.plot(kind='bar', figsize=(14, 7), width=0.8)
plt.title('Actual vs Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()

# Plotting predictions vs actual using a histogram plot
plt.figure(figsize=(14, 7))

# Plot histogram for actual returns
plt.hist(y_test, bins=30, alpha=0.5, label='Actual Returns', color='b')

# Plot histogram for predicted returns
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Returns', color='r')

plt.title('Actual vs Predicted Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plotting predictions vs actual using a scatter plot
plt.figure(figsize=(14, 7))

# Scatter plot for actual vs predicted returns
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual', color='b')

# Plotting the diagonal line (ideal prediction line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Prediction')

plt.title('Actual vs Predicted Returns')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.legend()
plt.show()

# Plotting predictions vs actual using a radar plot
# Select a subset of dates for the radar plot
subset_size = 10
subset_dates = y_test.index[:subset_size]
subset_actual = y_test.values[:subset_size]
subset_predicted = y_pred[:subset_size]

# Number of variables
num_vars = len(subset_dates)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The radar plot requires the data to be closed
subset_actual = np.concatenate((subset_actual, [subset_actual[0]]))
subset_predicted = np.concatenate((subset_predicted, [subset_predicted[0]]))
angles += angles[:1]

# Plot the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.fill(angles, subset_actual, color='blue', alpha=0.25, label='Actual Returns')
ax.plot(angles, subset_actual, color='blue', linewidth=2)

ax.fill(angles, subset_predicted, color='red', alpha=0.25, label='Predicted Returns')
ax.plot(angles, subset_predicted, color='red', linewidth=2)

# Add the labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels([str(date.date()) for date in subset_dates])

# Add a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.title('Actual vs Predicted Returns (Radar Plot)')
plt.show()

# Portfolio Optimization
from scipy.optimize import minimize

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_returns - risk_free_rate) / p_std

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

mean_returns = returns.mean()
cov_matrix = returns.cov()
optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)

optimal_weights = optimal_portfolio['x']
print(f'Optimal Weights: {optimal_weights}')

def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std
    return results, weights_record

# Assuming mean_returns and cov_matrix are already defined
results, weights_record = plot_efficient_frontier(mean_returns, cov_matrix)

plt.figure(figsize=(10, 7))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar(label='Sharpe Ratio')
plt.title('Efficient Frontier')
plt.show()

plot_efficient_frontier(mean_returns, cov_matrix)
