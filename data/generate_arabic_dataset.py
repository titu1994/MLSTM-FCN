import numpy as np
from scipy.io import loadmat

arabic_path = r"../data/arabic/"

''' Load train set '''
arabic_training_dict = loadmat(arabic_path + "training_set_arabic.mat")
X = arabic_training_dict['training_set']
y = arabic_training_dict['train_labels']

X = X[0]

max_sequence_length = -np.inf
min_sequence_length = np.inf

for i in range(X.shape[0]):
    var_count = X[i].shape[-1]

    if var_count > max_sequence_length:
        max_sequence_length = var_count

    if var_count < min_sequence_length:
        min_sequence_length = var_count

print('max sequence length : ', max_sequence_length)
print('min sequence length : ', min_sequence_length)

X_train = np.zeros((X.shape[0], X[0].shape[0], max_sequence_length))
y_train = y[0]

# pad ending with zeros to get numpy arrays
for i in range(X_train.shape[0]):
    var_count = X[i].shape[-1]
    X_train[i, :, :var_count] = X[i]

''' Load test set '''
arabic_test_dict = loadmat(arabic_path + "test_set_arabic.mat")
print(arabic_test_dict.keys())
X = arabic_test_dict['test_set']
y = arabic_test_dict['test_labels']

X = X[0]
y = y[0]

X_test = np.zeros((X.shape[0], X[0].shape[0], max_sequence_length))
y_test = y

# pad ending with zeros to get numpy arrays
for i in range(X_test.shape[0]):
    var_count = X[i].shape[-1]
    X_test[i, :, :var_count] = X[i]

''' Save the datasets '''

np.save(arabic_path + 'X_train.npy', X_train)
np.save(arabic_path + 'y_train.npy', y_train)
np.save(arabic_path + 'X_test.npy', X_test)
np.save(arabic_path + 'y_test.npy', y_test)