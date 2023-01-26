import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
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

# Select a tree from the forest to plot
estimator = rf.estimators_[0]

# Plot the tree
plot_tree(estimator, filled=True)
plt.show()
