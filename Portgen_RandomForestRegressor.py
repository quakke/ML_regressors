from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time

X = pd.read_csv('test_data_ml/portgen/X.csv')
Y = pd.read_csv('test_data_ml/portgen/Y.csv')
X_test = pd.read_csv('test_data_ml/portgen/test_x.csv')
Y_test = pd.read_csv('test_data_ml/portgen/test_y.csv')
count = pd.read_csv('test_data_ml/portgen/count_995.csv')

start_time = time.time()

randomForest = RandomForestRegressor(
    bootstrap=True, criterion='mse', max_depth=500,
    max_features='auto', max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
    oob_score=False, random_state=0, verbose=0, warm_start=False)

randomForest.fit(X, Y.values.ravel());

predictions = randomForest.predict(X_test)
errors = abs(predictions - Y_test.values.ravel())

print('Mean Absolute Error for RF:', round(np.mean(errors), 4), 'degrees.')
print (np.mean(cross_val_score(randomForest, X_test, Y_test.values.ravel(), cv=10)))
print("RF portgen --- %s seconds ---" % (time.time() - start_time))

fig, ax = plt.subplots()
ax.scatter(count, predictions, c='red')
plt.scatter(count, predictions, c='red')
ax.scatter(count, Y_test, c='blue')
plt.scatter(count, Y_test, c='blue')
plt.xlabel('predictions, сек.',fontsize=12)
plt.ylabel('real, сек.',fontsize=12)
plt.legend(fontsize=13,loc=4)
plt.title("portgen RF")
plt.show()