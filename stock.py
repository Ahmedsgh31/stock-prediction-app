import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

# Title and Description
st.title("Stock Price Prediction with AI")
st.write("Enter the stock ticker symbol to fetch data from Yahoo Finance, visualize trends, and predict future stock prices using an optimized LSTM model.")

# Input for Stock Ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):")

if stock_ticker:
    try:
        # Fetch Data from Yahoo Finance
        stock_data = yf.download(stock_ticker, start="2010-01-01", end=None)
        stock_data.reset_index(inplace=True)

        # Fetch Current Stock Price
        ticker_info = yf.Ticker(stock_ticker)
        current_price = ticker_info.history(period="1d")['Close'].iloc[-1]

        # Display Current Price
        st.subheader(f"Current Stock Price for {stock_ticker}: ${current_price:.2f}")

        # Preprocess Data
        data = stock_data[['Date', 'Close']].dropna()  # Drop rows with missing values
        data = data.rename(columns={"Date": "ds", "Close": "y"})
        data['ds'] = pd.to_datetime(data['ds'])

        # Prepare Data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['y']])

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(dataset, look_back=60):
            X, y = [], []
            for i in range(look_back, len(dataset)):
                X.append(dataset[i-look_back:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        look_back = 60
        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build the Optimized LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))

        # Compile and Train the Model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        st.subheader("Model Training (Optimized)")
        with st.spinner("Training the optimized LSTM model..."):
            model.fit(X_train, y_train, batch_size=32, epochs=20)  # Reduced epochs

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_scaled = scaler.inverse_transform([y_test])

        # Add a table to compare the last 30 days actual vs prediction
        st.subheader("Last 30 Days: Actual vs Predicted Prices")
        last_30_dates = data['ds'][-len(predictions):][-30:]
        actual_vs_predicted_df = pd.DataFrame({
            "Date": last_30_dates,
            "Actual Price": y_test_scaled[0][-30:],
            "Predicted Price": predictions[:, 0][-30:]
        })
        st.write(actual_vs_predicted_df)

        # Prediction Accuracy Metrics
        st.subheader("Prediction Accuracy Metrics (Last 30 Days)")
        mape = mean_absolute_percentage_error(actual_vs_predicted_df["Actual Price"], actual_vs_predicted_df["Predicted Price"])
        rmse = np.sqrt(mean_squared_error(actual_vs_predicted_df["Actual Price"], actual_vs_predicted_df["Predicted Price"]))

        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2%}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

        # Plot Predictions (Last 30 Days)
        st.subheader("Last 30 Days: Predicted vs. Actual Stock Prices")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=actual_vs_predicted_df["Date"], y=actual_vs_predicted_df["Actual Price"],
                                  mode='lines', name='Actual'))
        fig2.add_trace(go.Scatter(x=actual_vs_predicted_df["Date"], y=actual_vs_predicted_df["Predicted Price"],
                                  mode='lines', name='Predicted'))
        fig2.update_layout(
            title="Predicted vs. Actual Stock Prices (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig2)

        # Plot Actual vs Forecast for the Last 5 Years
        st.subheader("Last 5 Years: Actual vs Forecasted Prices")
        last_5_years_dates = data['ds'][-len(predictions):]
        actual_vs_forecast_df = pd.DataFrame({
            "Date": last_5_years_dates,
            "Actual Price": y_test_scaled[0],
            "Forecasted Price": predictions[:, 0]
        })

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=actual_vs_forecast_df["Date"], y=actual_vs_forecast_df["Actual Price"],
                                  mode='lines', name='Actual'))
        fig3.add_trace(go.Scatter(x=actual_vs_forecast_df["Date"], y=actual_vs_forecast_df["Forecasted Price"],
                                  mode='lines', name='Forecasted'))
        fig3.update_layout(
            title="Actual vs Forecasted Prices (Last 5 Years)",
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig3)

        # Future Prediction
        st.subheader("Next 30 Days Prediction")
        last_60_days = scaled_data[-look_back:].reshape(-1, 1)
        X_future = last_60_days.reshape(1, look_back, 1)

        future_predictions = []
        future_dates = []
        current_date = data['ds'].iloc[-1]

        while len(future_predictions) < 30:
            future_price = model.predict(X_future)  # Predict next price
            future_predictions.append(future_price[0, 0])
            # Append the predicted price to the input data for the next prediction
            future_price = np.array(future_price).reshape(1, 1, 1)
            X_future = np.append(X_future[:, 1:, :], future_price, axis=1)

            # Add next date skipping weekends
            current_date += pd.Timedelta(days=1)
            while current_date.weekday() >= 5:  # Skip Saturday and Sunday
                current_date += pd.Timedelta(days=1)
            future_dates.append(current_date)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
        st.write(future_df)

        # Plot Future Predictions
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'], mode='lines', name='Future Prediction'))
        fig4.update_layout(
            title="30-Day Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Price",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig4)

    except Exception as e:
        st.error(f"Failed to fetch or process data for ticker {stock_ticker}. Error: {str(e)}")
