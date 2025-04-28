import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚗",
    layout="wide"
)

# Add a title and description
st.title("🚗 Traffic Volume Prediction App")
st.write("""
This application predicts traffic volume based on historical data. 
Enter a date to see the predicted traffic volume for that day.
""")

@st.cache_data
def load_data():
    """Load and preprocess the traffic volume dataset"""
    try:
        df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
        df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_prophet_data(df):
    """Prepare data for Prophet model"""
    # Resample to daily data
    daily_data = df.set_index('date_time')['traffic_volume'].resample('D').mean()
    daily_data = daily_data.fillna(daily_data.bfill())
    
    # Create DataFrame for Prophet
    prophet_data = pd.DataFrame({
        'ds': daily_data.index,
        'y': daily_data.values
    })
    return prophet_data

def train_prophet_model(prophet_data):
    """Train the Prophet model"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_data)
    return model

def predict_traffic(model, prediction_date):
    """Make predictions for the specified date"""
    future_df = pd.DataFrame({'ds': [prediction_date]})
    forecast = model.predict(future_df)
    return forecast

def plot_traffic_forecast(model, days=30):
    """Create a forecast plot for the next specified days"""
    # Create future dates for prediction
    future_dates = pd.DataFrame({
        'ds': pd.date_range(start=datetime.now(), periods=days, freq='D')
    })
    
    # Generate the forecast
    forecast = model.predict(future_dates)
    
    # Plot using Plotly
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='blue')
    ))
    
    # Add upper and lower bounds
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 255, 0.2)',
        fill='tonexty',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Traffic Volume Forecast',
        xaxis_title='Date',
        yaxis_title='Traffic Volume',
        hovermode='x'
    )
    
    return fig

def plot_components(model, forecast):
    """Plot the forecast components"""
    fig = model.plot_components(forecast)
    return fig

def main():
    # Load data
    df = load_data()
    
    if df is not None:
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Dataset Overview", "Prediction", "Model Insights"])
        
        if page == "Dataset Overview":
            st.header("Dataset Overview")
            
            # Display basic information about the dataset
            st.subheader("First few rows of the dataset")
            st.write(df.head())
            
            st.subheader("Dataset Information")
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Time range: {df['date_time'].min().date()} to {df['date_time'].max().date()}")
            
            # Traffic volume distribution
            st.subheader("Traffic Volume Distribution")
            fig = px.histogram(df, x="traffic_volume", nbins=50,
                              title="Distribution of Traffic Volume")
            st.plotly_chart(fig)
            
            # Traffic by hour of day
            st.subheader("Traffic by Hour of Day")
            hourly_traffic = df.groupby(df['date_time'].dt.hour)['traffic_volume'].mean().reset_index()
            fig = px.line(hourly_traffic, x='date_time', y='traffic_volume',
                         title="Average Traffic Volume by Hour")
            fig.update_xaxes(title="Hour of Day")
            st.plotly_chart(fig)
            
            # Traffic by day of week
            st.subheader("Traffic by Day of Week")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df['day_of_week'] = df['date_time'].dt.dayofweek
            day_traffic = df.groupby('day_of_week')['traffic_volume'].mean().reset_index()
            day_traffic['day_name'] = day_traffic['day_of_week'].apply(lambda x: days[x])
            fig = px.bar(day_traffic, x='day_name', y='traffic_volume',
                        title="Average Traffic Volume by Day of Week")
            fig.update_xaxes(title="Day of Week")
            st.plotly_chart(fig)


            # Traffic by holiday
            st.subheader("Traffic Volume on Holidays vs Non-Holidays")

            # Group by holiday status and calculate mean traffic
            holiday_traffic = df.groupby('holiday')['traffic_volume'].mean().reset_index()

            # Create interactive bar plot
            fig = px.bar(holiday_traffic, 
                        x='holiday', 
                        y='traffic_volume',
                        labels={'holiday': 'Holiday Status', 'traffic_volume': 'Average Traffic Volume'},
                        title="Average Traffic Volume: Holiday vs Regular Days",
                        color='holiday')

            # Customize layout
            fig.update_layout(showlegend=False,
                            xaxis={'categoryorder':'total descending'})
            fig.update_xaxes(tickvals=[0, 1], ticktext=['Non-Holiday', 'Holiday'])
            # Display in Streamlit
            st.plotly_chart(fig)


            # Group by weather condition and calculate mean traffic volume
            weather_traffic = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False).reset_index()

            st.subheader("Average Traffic Volume by Weather Condition")

            # Create interactive bar plot with Plotly
            fig = px.bar(
                weather_traffic,
                x='weather_main',
                y='traffic_volume',
                labels={'weather_main': 'Weather Condition', 'traffic_volume': 'Average Traffic Volume'},
                title='Average Traffic Volume by Weather Condition'
            )
            fig.update_layout(xaxis_tickangle=-45)

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
        elif page == "Prediction":
            st.header("Traffic Volume Prediction")
            
            # Prepare data for Prophet
            prophet_data = prepare_prophet_data(df)
            
            # Train the model
            with st.spinner('Training the Prophet model...'):
                model = train_prophet_model(prophet_data)
            
            # Date input for prediction
            st.subheader("Select a Date for Prediction")
            prediction_date = st.date_input(
                "Choose a date", 
                value=datetime.now() + timedelta(days=1),
                min_value=datetime.now()
            )
            
            # Make prediction
            if st.button("Predict Traffic Volume"):
                with st.spinner('Generating prediction...'):
                    forecast = predict_traffic(model, prediction_date)
                    
                    # Display prediction results
                    st.subheader("Prediction Results")
                    
                    # Create a metrics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Traffic Volume", f"{int(forecast['yhat'].values[0])}")
                    with col2:
                        st.metric("Lower Bound", f"{int(forecast['yhat_lower'].values[0])}")
                    with col3:
                        st.metric("Upper Bound", f"{int(forecast['yhat_upper'].values[0])}")
                    
                    # Plot forecast for the next 30 days
                    st.subheader("30-Day Forecast")
                    forecast_fig = plot_traffic_forecast(model)
                    st.plotly_chart(forecast_fig)
                    
                    # Plot forecast components
                    st.subheader("Forecast Components")
                    with st.spinner('Generating component plots...'):
                        # Create a future dataframe for the components
                        future = model.make_future_dataframe(periods=30)
                        full_forecast = model.predict(future)
                        
                        # Plot trends and seasonality
                        fig = model.plot_components(full_forecast)
                        st.pyplot(fig)
            
        elif page == "Model Insights":
            st.header("Model Insights")
            
            st.subheader("About the Prediction Model")
            st.write("""
            This app uses Facebook's Prophet model for time series forecasting. The model was selected because:
            
            1. It handles seasonal patterns well, which are important in traffic prediction
            2. It can incorporate holiday effects
            3. It's robust to missing data and outliers
            4. It automatically detects changepoints in the time series
            
            The model achieved the lowest RMSE (Root Mean Square Error) of 357.56 compared to other models like ARIMA and SARIMA.
            """)
            
            st.subheader("Key Performance Metrics")
            metrics = pd.DataFrame({
                'Model': ['ARIMA', 'SARIMA', 'Prophet'],
                'MSE': [221238.26, 134638.16, 127846.81],
                'RMSE': [470.36, 366.93, 357.56],
                'MAE': [391.84, 262.73, 256.95]
            })
            st.table(metrics)
            
            st.subheader("Factors Affecting Traffic Volume")
            st.write("""
            Based on the analysis, the following factors significantly affect traffic volume:
            
            - **Time of day**: Peak hours are typically 3 PM - 5 PM
            - **Day of week**: Weekdays have higher volumes than weekends, with Thursday and Friday being the busiest
            - **Weather conditions**: Severe weather like "Squall" results in decreased traffic volume
            - **Holidays**: Traffic volume is generally lower on holidays
            """)

if __name__ == "__main__":
    main()
