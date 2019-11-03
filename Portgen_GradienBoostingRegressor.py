from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.model_selection import cross_val_score
import time

X = pd.read_csv('test_data_ml/portgen/X.csv')
Y = pd.read_csv('test_data_ml/portgen/Y.csv')
X_test = pd.read_csv('test_data_ml/portgen/test_x.csv')
Y_test = pd.read_csv('test_data_ml/portgen/test_y.csv')
count = pd.read_csv('test_data_ml/portgen/count_995.csv')

start_time = time.time()

gradientBoosting = GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=1000, max_features='log2', min_samples_split=200, max_depth=500)
gradientBoosting.fit(X, Y.values.ravel());

predictions = gradientBoosting.predict(X_test)
errors = abs(predictions - Y_test.values.ravel())

print("GB portgen --- %s seconds ---" % (time.time() - start_time))
print('Mean Absolute Error for GradientBoosting:', round(np.mean(errors), 4), 'degrees.')
print (np.mean(cross_val_score(gradientBoosting, X_test, Y_test.values.ravel(), cv=10)))

fig, ax = plt.subplots()
ax.scatter(count, predictions, c='black')
plt.scatter(count, predictions, c='black')
ax.scatter(count, Y_test, c='blue')
plt.scatter(count, Y_test, c='blue')
plt.xlabel('predictions, сек.',fontsize=12)
plt.ylabel('real, сек.',fontsize=12)
plt.legend(fontsize=13,loc=4)
plt.title("portgen GB")
plt.show()