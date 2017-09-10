import numpy as np
np.random.seed(1000)

from scipy.io import loadmat

ht_sensor_path = r""

''' Load train set '''
data_dict = loadmat(ht_sensor_path + "HT_Sensor_dataset.mat")
X_train_mat = data_dict['X_train'][0]
y_train_mat = data_dict['Y_train'][0]
X_test_mat = data_dict['X_test'][0]
y_test_mat = data_dict['Y_test'][0]

print(X_train_mat.shape, X_test_mat.shape)

y_train = y_train_mat.reshape(-1, 1)
y_test = y_test_mat.reshape(-1, 1)

var_list = []
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
max_nb_timesteps = var_list.max()
min_nb_timesteps = var_list.min()
median_nb_timesteps = int(np.median(var_list))

print('max nb timesteps train : ', max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median nb timesteps train : ', median_nb_timesteps)

''' 
We use the min_nb_timesteps in this dataset as our number of variables,
since the maximum is nearly 15000 timesteps and cannot be trained due to
GPU memory constraints.

However, comparison of models on this dataset should be against either
full (max_nb_timesteps) or limited (min_nb_timesteps). 

Baseline provided for limited (min_nb_timesteps) data only.
'''
X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], min_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_count = min(var_count, min_nb_timesteps)
    X_train[i, :, :var_count] = X_train_mat[i][:, :min_nb_timesteps]

# ''' Load test set '''

X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], min_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_test_mat.shape[0]):
    var_count = X_test_mat[i].shape[-1]

    # skip empty tuples
    if var_count == 0:
        continue

    var_count = min(var_count, min_nb_timesteps)
    X_test[i, :, :var_count] = X_test_mat[i][:, :min_nb_timesteps]


# ''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

np.save(ht_sensor_path + 'X_train.npy', X_train)
np.save(ht_sensor_path + 'y_train.npy', y_train)
np.save(ht_sensor_path + 'X_test.npy', X_test)
np.save(ht_sensor_path + 'y_test.npy', y_test)
