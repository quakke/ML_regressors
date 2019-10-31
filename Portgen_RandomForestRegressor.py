from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time

X = pd.read_csv('/Users/quak/Documents/test_data_ml/portgen/X.csv')
Y = pd.read_csv('/Users/quak/Documents/test_data_ml/portgen/Y.csv')
X_test = pd.read_csv('/Users/quak/Documents/test_data_ml/portgen/test_x.csv')
Y_test = pd.read_csv('/Users/quak/Documents/test_data_ml/portgen/test_y.csv')
count = pd.read_csv('/Users/quak/Documents/test_data_ml/portgen/count_995.csv')

start_time = time.time()

randomForest = RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=1000, min_samples_leaf=4)
randomForest.fit(X, Y.values.ravel());

predictions = randomForest.predict(X_test)
errors = abs(predictions - Y_test.values.ravel())

print('Mean Absolute Error for RF:', round(np.mean(errors), 4), 'degrees.')
print (np.mean(cross_val_score(randomForest, X_test, Y_test.values.ravel(), cv=10)))
print("RF portgen --- %s seconds ---" % (time.time() - start_time))

fig, ax = plt.subplots()
ax.scatter(predictions, count, c='black')
plt.scatter(predictions, count, c='black')
ax.scatter(Y_test, count, c='black')
plt.scatter(Y_test, count, c='black')
plt.xlabel('predictions, сек.',fontsize=12)
plt.ylabel('real, сек.',fontsize=12)
plt.legend(fontsize=13,loc=4)
plt.title("portgen RF")
plt.show()