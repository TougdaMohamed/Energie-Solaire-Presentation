import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load your data into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# extract the features and target variables
X = df[['Temp_Avg', 'RH_Avg', 'WSpeed_Avg', 'WSpeed_Max', 'WDir_Avg', 'Rain_Tot', 'Press_Avg']]
y = df['Rad_Avg']

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize the model
rf = RandomForestRegressor(n_estimators=100)

# train the model on the training data
rf.fit(X_train, y_train)

# make predictions on the test data
y_pred = rf.predict(X_test)

# evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))




# create the scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual solar radiation')
plt.ylabel('Predicted solar radiation')
plt.title('Random Forest Regression')
plt.show()



# Select a tree from the forest to plot
estimator = rf.estimators_[0]

# Plot the tree
plot_tree(estimator, filled=True)
plt.show()


import matplotlib.pyplot as plt

# scatter plot of actual vs predicted solar radiation
plt.scatter(y_test, y_pred)

# line of best fit (linear regression model)
plt.plot(y_test, y_test, color='red')

plt.xlabel('Actual Solar Radiation (w/m^2)')
plt.ylabel('Predicted Solar Radiation (w/m^2)')
plt.title('Regression Plot for Actual vs Predicted Solar Radiation')
plt.show()

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------


import matplotlib.pyplot as plt
import pandas as pd

# load your data into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# extract the features and target variables
X = df[['Temp_Avg', 'RH_Avg', 'WSpeed_Avg', 'WSpeed_Max', 'WDir_Avg', 'Rain_Tot', 'Press_Avg']]
y = df['Rad_Avg']

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize the model
rf = RandomForestRegressor(n_estimators=100)

# train the model on the training data
rf.fit(X_train, y_train)

# make predictions on the test data
y_pred = rf.predict(X_test)

# create a dataframe to store the actual and predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# select range of 70 hrs
df = df.head(70)

# plot the actual and predicted values
df.plot(kind='line',figsize=(10,8))
plt.title('Regression Plot of Actual and Predicted Solar Radiation (W/m2) in function of time')
plt.xlabel('Time (in hrs)')
plt.ylabel('Solar Radiation (W/m2)')
plt.show()






import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load your data into a DataFrame
df = pd.read_csv('cleaned_data.csv', parse_dates=[0])
df.set_index('TIMESTAMP', inplace=True)

# extract the features and target variables
X = df[['Temp_Avg', 'RH_Avg', 'WSpeed_Avg', 'WSpeed_Max', 'WDir_Avg', 'Rain_Tot', 'Press_Avg']]
y = df['Rad_Avg']

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize the model
rf = RandomForestRegressor(n_estimators=100)

# train the model on the training data
rf.fit(X_train, y_train)

# make predictions on the test data
y_pred = rf.predict(X_test)

# Extract the timestamp for the first 70 hours in the test dataset
df_test = X_test.iloc[:70, :]
df_test['Actual_Radiation'] = y_test[:70]
df_test['Predicted_Radiation'] = y_pred[:70]

# plot the actual and predicted solar radiation against the timestamp
plt.plot(df_test.index, df_test['Actual_Radiation'], label='Actual')
plt.plot(df_test.index, df_test['Predicted_Radiation'], label='Predicted')
plt.xlabel('Time (in hrs)')
plt.ylabel('Solar Radiation (W/m2)')
plt.show()

# -----------------------------------------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------