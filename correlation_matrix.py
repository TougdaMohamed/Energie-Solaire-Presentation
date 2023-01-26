import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load your data into a DataFrame
df = pd.read_csv('cleaned_data.csv')

# extract the features and target variables
X = df[['Temp_Avg', 'RH_Avg', 'WSpeed_Avg', 'WSpeed_Max', 'WDir_Avg', 'Rain_Tot', 'Press_Avg']]
y = df['Rad_Avg']

# concatenate the feature and target variables into a single dataframe
data = pd.concat([X, y], axis=1)

# calculate the correlation matrix
corr_matrix = data.corr()

# display the correlation matrix
print(corr_matrix)




# create a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True)

# show the plot
plt.show()
