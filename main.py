# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 2: Load dataset
df = pd.read_csv("pharmacy_dataset.csv")

print("First 5 rows:")
print(df.head())

# Step 3: Data Cleaning
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Step 4: Feature Engineering
df = df.sort_values('Date')

df['Time_Index'] = range(len(df))
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Added previous sales 
df['Prev_Sales'] = df['Quantity_Sold'].shift(1)
df.dropna(inplace=True)

# Step 5: Basic Analysis
top_meds = df.groupby('Medicine_Name')['Quantity_Sold'].sum().sort_values(ascending=False).head(5)
print("\nTop Selling Medicines:\n", top_meds)

low_meds = df.groupby('Medicine_Name')['Quantity_Sold'].sum().sort_values().head(5)
print("\nLow Selling Medicines:\n", low_meds)

df['Revenue'] = df['Quantity_Sold'] * df['Price']
print("\nTotal Revenue:", df['Revenue'].sum())

# Graph 1
top_meds.plot(kind='bar')
plt.title("Top Selling Medicines")
plt.ylabel("Quantity Sold")
plt.show()

# Graph 2
df.groupby('Date')['Quantity_Sold'].sum().plot()
plt.title("Sales Trend Over Time")
plt.ylabel("Quantity Sold")
plt.show()

# Step 6: Preparation of data for ML
df['Category'] = df['Category'].astype('category').cat.codes

X = df[['Price', 'Category', 'Month', 'Discount', 'Time_Index', 'Prev_Sales']]
y = df['Quantity_Sold']

# Time-based split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 7: Training Model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
mae = mean_absolute_error(y_test, y_pred)
print("\nModel MAE:", mae)

# Step 10: Plot Actual vs Predicted
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()