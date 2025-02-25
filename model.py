#Impor libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Data reading and processing
df = pd.read_csv('dataset/Advertising_Budget_and_Sales.csv')
X = df.iloc[: , 1:-1].values
y = df.iloc[:, -1].values

#Splits  data into training and testing sets
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print(X_test)
print(y_test)
#Load the model and predict
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})

# Display the first 10 rows to compare line by line
print(comparison_df.head(20))

# Calculating error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

# Define the new campaign's advertising budgets
new_campaign_data = {
    'TV Ad Budget ($)': [25],        # TV advertising budget in thousands of dollars
    'Radio Ad Budget ($)': [11],     # Radio advertising budget in thousands of dollars
    'Newspaper Ad Budget ($)': [29]  # Newspaper advertising budget in thousands of dollars
}

# Convert the dictionary to a DataFrame
new_campaign_df = pd.DataFrame(new_campaign_data)

# Display the DataFrame to verify
print(new_campaign_df)

# Predict the sales using the trained model
predicted_sales = regressor.predict(new_campaign_df)

# Display the predicted sales
print(f"Predicted Sales for the New Campaign: {predicted_sales[0]}")





