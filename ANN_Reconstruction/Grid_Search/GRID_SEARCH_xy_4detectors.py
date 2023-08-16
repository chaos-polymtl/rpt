# ============================================================================
# Grid search with cross validation to find the hyperparameters of the model.
# Author : Maxime Landry, Polytechnique Montréal, 2022
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# ---------------------------------------------------------------------------

# Read the data
pd_dat = pd.read_csv('Feed.txt', delimiter='\t')

# Extract the values from the dataframe
dataset = pd_dat.values

#Normalizing the input counts
X_raw=dataset[:,:4]
scaler_X = MinMaxScaler()
scaler_X.fit(X_raw)
X_scale= scaler_X.transform(X_raw)


X_train_x, X_test_x, Y_train_x, Y_test_x = train_test_split(X_scale[:,:4],dataset[:,4], test_size=0.2)
X_train_y, X_test_y, Y_train_y, Y_test_y = train_test_split(X_scale[:,:4],dataset[:,5], test_size=0.2)


# Grid search
def create_model(neurons=1, layers=1, activation='elu', optimizer='adamax'):
    model = Sequential()
    layer = 0
    while layer < layers:
        model.add(Dense(neurons, input_dim=4, kernel_initializer=keras.initializers.GlorotUniform(), activation=activation))
        layer += 1
    model.add(Dense(1, kernel_initializer=keras.initializers.GlorotUniform(), activation='linear'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model

seed = 4
np.random.seed(seed)
model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=2)
epochs = [3000]
batch_size = [20000]
neurons = [70,90,110]
layers = [5,6]
param_grid = dict(epochs=epochs,
                  batch_size=batch_size,
                  neurons=neurons,
                  layers=layers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_root_mean_squared_error')

grid_result_x = grid.fit(X_train_x, Y_train_x)

# Summarize results x
means = grid_result_x.cv_results_['mean_test_score']
stds = grid_result_x.cv_results_['std_test_score']
params = grid_result_x.cv_results_['params']
for i in range(len(params)):
    params[i]['mean'] = means[i]
    params[i]['std'] = stds[i]
df = pd.DataFrame(params)
df.to_excel('grid_search_x.xlsx')
print("Best: %f using %s" % (grid_result_x.best_score_, grid_result_x.best_params_))

grid_result_y = grid.fit(X_train_y, Y_train_y)

# Summarize results y
means = grid_result_y.cv_results_['mean_test_score']
stds = grid_result_y.cv_results_['std_test_score']
params = grid_result_y.cv_results_['params']
for i in range(len(params)):
    params[i]['mean'] = means[i]
    params[i]['std'] = stds[i]
df = pd.DataFrame(params)
df.to_excel('grid_search_y.xlsx')
print("Best: %f using %s" % (grid_result_y.best_score_, grid_result_y.best_params_))
