import numpy as np
from scipy.io import loadmat

arabic_path = r"../data/arabic_voice/"

''' Load train set '''
DATA = loadmat(arabic_path + "arabic_voice_window_3_ifperm_1.mat")

X_data = DATA['new_X'][0]
y_data = DATA['new_labels'][0]

labels_n = 88
each_label_number = 100

pre = 0
train_ind = []
test_ind = []
training_size = 75
test_size = 25

for i in range (1,(labels_n+1)):
    ti = list(range(pre+1,pre+training_size+1))
    train_ind.append(ti)
    test_ind.append(list(range(pre+training_size+1,pre+training_size+test_size+1)))
    pre = pre + each_label_number


trainind = [item for sublist in train_ind for item in sublist]
testind = [item for sublist in test_ind for item in sublist]

''' Load train set '''
X = X_data[np.array(trainind)-1] ###Train data

max_nb_variables = -np.inf
min_nb_variables = np.inf

for i in range(X.shape[0]):
    var_count = X[i].shape[-1]

    if var_count > max_nb_variables:
        max_nb_variables = var_count

    if var_count < min_nb_variables:
        min_nb_variables = var_count

print('max number variables : ', max_nb_variables)
print('min number variables : ', min_nb_variables)

X_train = np.zeros((X.shape[0], X[0].shape[0], max_nb_variables))
y_train = y_data[(np.array(trainind)-1)] ###Train Labels

# pad ending with zeros to get numpy arrays
for i in range(X_train.shape[0]):
    var_count = X[i].shape[-1]
    X_train[i, :, :var_count] = X[i]

''' Load test set '''

X = X_data[(np.array(testind)-1)] ####Test Data

X_test = np.zeros((X.shape[0], X[0].shape[0], max_nb_variables))
y_test = y_data[(np.array(testind)-1)] ####Test Labels​

# pad ending with zeros to get numpy arrays
for i in range(X_test.shape[0]):
    var_count = X[i].shape[-1]
    X_test[i, :, :var_count] = X[i]

''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Nb classes : ", len(np.unique(y_train)))

np.save(arabic_path + 'X_train.npy', X_train)
np.save(arabic_path + 'y_train.npy', y_train)
np.save(arabic_path + 'X_test.npy', X_test)
np.save(arabic_path + 'y_test.npy', y_test)
