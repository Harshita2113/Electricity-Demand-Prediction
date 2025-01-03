import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
import holidays

# Add a title and heading for the page
st.title("Electricity Demand Prediction Dashboard")
st.header("Predicting Electricity Demand for the Next Few Days")

# Load the dataset
file_path = 'Final 2023.csv'
data = pd.read_csv(file_path)

# Convert 'datetime' column to datetime type
data['datetime'] = pd.to_datetime(data['datetime'])

# Encode categorical variables
le = LabelEncoder()
data['Public Holiday'] = le.fit_transform(data['Public Holiday'])

# Feature Engineering
data['day_of_week'] = data['datetime'].dt.dayofweek
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour

# Select relevant features for the model
features = data[['temp', 'humidity', 'windgust', 'sealevelpressure', 'cloudcover', 
                 'visibility', 'solarradiation', 'uvindex', 'day_of_week', 'month', 
                 'hour', 'Public Holiday']]

target = data['Hourly Demand Met (in MW)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a model (using RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R² score (model accuracy)
r2 = r2_score(y_test, y_pred)

# Display the model performance metrics
st.write(f"Model Mean Absolute Error: {mae:.2f} MW")
st.write(f"Model R² Score (Accuracy): {r2:.2f}")

# Get public holidays for India using the holidays package
india_holidays = holidays.India(years=2024)

# Define a function to check if a date is a public holiday or weekend
def is_public_holiday_or_weekend(date):
    return 1 if date.date() in india_holidays or date.weekday() >= 5 else 0

# User input fields in the sidebar
with st.sidebar:
    
    num_days = st.slider("Select the number of days to predict electricity demand (max 15 days):", min_value=1, max_value=15, value=7)

# Predict the user-specified number of days starting from the current date
current_date = pd.Timestamp.now().normalize()  # Get the current date and normalize to midnight
future_dates = pd.date_range(start=current_date, periods=num_days*24, freq='H')

# Prepare future data for prediction
future_data = pd.DataFrame({
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month,
    'hour': future_dates.hour,
    'Public Holiday': [is_public_holiday_or_weekend(date) for date in future_dates]
})

# Add the weather-related features to future data
future_data['temp'] = data['temp'].mean()  # Assuming average temperature for simplicity
future_data['humidity'] = data['humidity'].mean()  # Average humidity
future_data['windgust'] = data['windgust'].mean()  # Average wind gust
future_data['sealevelpressure'] = data['sealevelpressure'].mean()  # Average pressure
future_data['cloudcover'] = data['cloudcover'].mean()  # Average cloud cover
future_data['visibility'] = data['visibility'].mean()  # Average visibility
future_data['solarradiation'] = data['solarradiation'].mean()  # Average solar radiation
future_data['uvindex'] = data['uvindex'].mean()  # Average UV index

# Ensure future_data has the same columns as the training data
future_data = future_data[features.columns]

# Make predictions for the future data
future_data['predicted_electricity_demand'] = model.predict(future_data)
future_data['date'] = future_dates.date

# Group by date to get daily sums of electricity demand
daily_data = future_data.groupby('date').agg({
    'predicted_electricity_demand': 'sum',
    'day_of_week': 'first',
    'Public Holiday': 'first'
}).reset_index()

# Map numerical day_of_week to actual day names
daily_data['day_name'] = daily_data['day_of_week'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})

# Map numerical Public Holiday to actual holiday status
daily_data['Public Holiday'] = daily_data['Public Holiday'].map({0: 'No', 1: 'Yes'})

# Update the date selection dropdown with available dates
with st.sidebar:
    date_options = daily_data['date'].tolist()
    selected_date = st.selectbox('Select a date to view hourly demand:', options=date_options)

# Display the daily predictions in the main area
st.write(f"Predicted Daily Electricity Demand for the Next {num_days} Days:")
st.dataframe(daily_data[['date', 'day_name', 'Public Holiday', 'predicted_electricity_demand']])

# Plot daily data using Matplotlib
fig, ax = plt.subplots()
ax.plot(daily_data['date'], daily_data['predicted_electricity_demand'], marker='o')
ax.set_title('Predicted Daily Electricity Demand')
ax.set_xlabel('Date')
ax.set_ylabel('Electricity Demand (MW)')
ax.grid(True)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)

# Display the date-wise plot
st.pyplot(fig)

# If a date is selected, display the hourly data and plot
if selected_date:
    # Filter the future_data for the selected date
    hourly_data = future_data[future_data['date'] == selected_date]

    # Display hourly demand for the selected date
    st.write(f"Hourly Demand for {selected_date}:")
    st.dataframe(hourly_data[['hour', 'predicted_electricity_demand']])

    # Plot hourly data using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(hourly_data['hour'], hourly_data['predicted_electricity_demand'], marker='o')
    ax.set_title(f'Hourly Electricity Demand on {selected_date}')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Electricity Demand (MW)')
    ax.grid(True)

    # Rotate x-axis labels by 45 degrees for hourly plot
    plt.xticks(rotation=45)

    # Display the hourly plot
    st.pyplot(fig)
