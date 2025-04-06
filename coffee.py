import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    df = pd.read_csv("./coffee_shop_revenue.csv")
    return df

def show_data_page(df):
    st.title("Data & Description")
    st.write("### Dataset Overview")
    st.write("- The dataset contains **2000 rows** and **7 columns** with no missing values.")
    st.write("- It tracks daily business metrics for a coffee shop.")
    
    st.write("### Column Descriptions")
    st.write("1. **Number_of_Customers_Per_Day** (*int64*): The number of customers visiting the shop each day.")
    st.write("2. **Average_Order_Value** (*float64*): The average amount spent per order (in dollars).")
    st.write("3. **Operating_Hours_Per_Day** (*int64*): The number of hours the shop operates daily.")
    st.write("4. **Number_of_Employees** (*int64*): The number of employees working each day.")
    st.write("5. **Marketing_Spend_Per_Day** (*float64*): Daily marketing expenditure (in dollars).")
    st.write("6. **Location_Foot_Traffic** (*int64*): The number of people passing by the shop.")
    st.write("7. **Daily_Revenue** (*float64*): The total revenue generated each day (in dollars).")
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Statistical Description")
    st.dataframe(df.describe().T)
def show_eda_page(df):
    st.title("Exploratory Data Analysis (EDA)")
    
    st.write("### Select Columns for Analysis")
    columns = st.multiselect("Choose one or more columns", df.columns, default=[df.columns[0]])
    
    st.write("### Correlation Matrix")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(fig)
    
    st.write("### Key Visualizations")
    for column in columns:
        if df[column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[column], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {column}")
            st.pyplot(fig)
        
    if len(columns) > 1:
        st.write("### Multi-Column Histograms")
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in columns:
            if df[column].dtype in ['int64', 'float64']:
                sns.histplot(df[column], bins=30, kde=True, ax=ax, label=column)
        ax.set_title("Multi-Column Histogram")
        ax.legend()
        st.pyplot(fig)
        
    for column in columns:
        if df[column].dtype in ['int64', 'float64']:
            st.write("### Scatter Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df[column], y=df['Daily_Revenue'], alpha=0.6, ax=ax)
            ax.set_title(f"{column} vs Daily Revenue")
            ax.set_xlabel(column)
            ax.set_ylabel("Daily Revenue ($)")
            st.pyplot(fig)
            
            st.write("### Box Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=df[column], y=df['Daily_Revenue'], ax=ax)
            ax.set_title(f"{column} vs Daily Revenue")
            ax.set_xlabel(column)
            ax.set_ylabel("Daily Revenue ($)")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=df[column], ax=ax)
            ax.set_title(f"Count of {column}")
            st.pyplot(fig)

def show_modeling_page(df):
    st.title("Time Series Modeling")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    st.write("### Time Series Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df["Daily_Revenue"], label="Daily Revenue", color="blue")
    plt.title("Daily Revenue Over Time")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    st.pyplot(fig)
    
    df['Revenue_7Day_MA'] = df['Daily_Revenue'].rolling(window=7).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df['Daily_Revenue'], label="Daily Revenue", alpha=0.5)
    plt.plot(df.index, df['Revenue_7Day_MA'], label="7-Day Moving Average", color="red")
    plt.title("7-Day Moving Average of Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    st.pyplot(fig)
    
    result = seasonal_decompose(df['Daily_Revenue'], model='additive', period=30)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    result.trend.plot(ax=axes[0], title='Trend')
    result.seasonal.plot(ax=axes[1], title='Seasonality')
    result.resid.plot(ax=axes[2], title='Residuals')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("### Autocorrelation of Revenue")
    fig, ax = plt.subplots(figsize=(10, 5))
    autocorrelation_plot(df['Daily_Revenue'])
    st.pyplot(fig)
    
    st.write("### Auto ARIMA Model Selection")
    arima_model = auto_arima(df['Daily_Revenue'], seasonal=False, trace=True, stepwise=True)
    st.text(f"Best ARIMA Model: {arima_model}")
    
    st.write("### ARIMA Model")
    arima = ARIMA(df['Daily_Revenue'], order=arima_model.order)
    arima_fit = arima.fit()
    st.text(arima_fit.summary())
    
    st.write("### SARIMA Model")
    sarima = SARIMAX(df['Daily_Revenue'], order=arima_model.order, seasonal_order=(1, 1, 1, 30))
    sarima_fit = sarima.fit()
    st.text(sarima_fit.summary())
    
    st.write("### Forecasting with SARIMA")
    forecast = sarima_fit.forecast(steps=30)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df['Daily_Revenue'], label="Historical Revenue", color="blue")
    plt.plot(pd.date_range(df.index[-1], periods=30, freq='D'), forecast, label="SARIMA Forecast", color="red")
    plt.title("Revenue Forecast (SARIMA Model)")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    st.pyplot(fig)
    
    st.write("### Forecasting with ARIMA")
    model_arima = auto_arima(df['Daily_Revenue'], seasonal=True, m=12, trace=True, suppress_warnings=True)
    model_arima.fit(df['Daily_Revenue'])
    
    last_date = df.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    
    date_range = pd.date_range(start=last_date, periods=(2040 - last_date.year) * 12, freq='M')
    forecast = model_arima.predict(n_periods=len(date_range))
    
    forecast_df = pd.DataFrame({'Date': date_range, 'Predicted_Daily_Revenue': forecast})
    forecast_df.set_index('Date', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df.index, df['Daily_Revenue'], label="Actual Revenue", color='blue')
    plt.plot(forecast_df.index, forecast_df['Predicted_Daily_Revenue'], label="Forecasted Revenue", color='red', linestyle='dashed')
    plt.xlabel("Year")
    plt.ylabel("Daily Revenue")
    plt.title("Forecasted Daily Revenue for 2040")
    plt.legend()
    st.pyplot(fig)
    
    st.write("### Machine Learning Modeling")
    features = ['Marketing_Spend_Per_Day', 'Number_of_Customers_Per_Day', 'Average_Order_Value', 'Number_of_Employees', 'Location_Foot_Traffic']
    X = df[features]
    y = df['Daily_Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=43),
        "XGBoost": XGBRegressor()
    }
    
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred) * 100  
        metrics.append([name, mae, mse, rmse, r2])
    
    metrics_df = pd.DataFrame(metrics, columns=["Model", "MAE", "MSE", "RMSE", "R² Score (%)"])
    metrics_df.set_index("Model", inplace=True)
    st.dataframe(metrics_df)
    st.write("### Model Performance Evaluation")
    metrics_df = pd.DataFrame(metrics, columns=["Model", "MAE", "MSE", "RMSE", "R² Score (%)"])
    metrics_df.set_index("Model", inplace=True)
    
    st.write("### RMSE Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=metrics_df.index, y=metrics_df["RMSE"], palette="Blues_r", ax=ax)
    ax.set_xlabel("ML Models")
    ax.set_ylabel("RMSE (Lower is Better)")
    ax.set_title("RMSE Comparison of ML Models")
    plt.xticks(rotation=30)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    st.pyplot(fig)
    
    st.write("### R² Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=metrics_df.index, y=metrics_df["R² Score (%)"], palette="Greens_r", ax=ax)
    ax.set_xlabel("ML Models")
    ax.set_ylabel("R² Score (%) (Higher is Better)")
    ax.set_title("R² Score Comparison of ML Models")
    plt.xticks(rotation=30)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    st.pyplot(fig)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data & Description", "EDA", "Modeling"])
    
    df = load_data()
    
    if page == "Data & Description":
        show_data_page(df)
    elif page == "EDA":
        show_eda_page(df)
    elif page == "Modeling":
        show_modeling_page(df)

if __name__ == "__main__":
    main()
