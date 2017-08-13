import scipy.io as sio
import numpy as np


def find1(a, func):
    qqq = [i for (i, val) in enumerate(a) if func(val)]
    return (np.array(qqq) + 1)


action_3d_path = r'../data/Action3D/'

DATA = sio.loadmat(action_3d_path + 'joint_feat_coordinate.mat')

feat = DATA['feat'][0]
if_contain = DATA['if_contain'][0]
labels = DATA['labels'][0]

data = feat

K = 20
train_ind = []
test_ind = []
testActors = [6, 7, 8, 9, 10]
i = 1
true_i = 0

for a in range(1, 21):
    for j in range(1, 11):
        for e in range(1, 4):
            if (if_contain[i - 1] == 0):
                i = i + 1
                continue
            true_i = true_i + 1
            if not (np.all((find1(testActors, lambda x: x == j)) == 0)):
                test_ind.append(true_i)
            else:
                train_ind.append(true_i)
            i = i + 1

''' Load train set '''
X = data[(np.array(train_ind) - 1)]

max_nb_variables = -np.inf
min_nb_variables = np.inf

for i in range(X.shape[0]):
    var_count = X[i].shape[-1]

    if var_count > max_nb_variables:
        max_nb_variables = var_count

    if var_count < min_nb_variables:
        min_nb_variables = var_count

print('max nb variables train : ', max_nb_variables)

X_train = np.zeros((X.shape[0], X[0].shape[0], max_nb_variables))
y_train = labels[(np.array(train_ind) - 1)]

# pad ending with zeros to get numpy arrays
for i in range(X_train.shape[0]):
    var_count = X[i].shape[-1]
    X_train[i, :, :var_count] = X[i]

''' Load test set '''
X = data[(np.array(test_ind) - 1)]

X_test = np.zeros((X.shape[0], X[0].shape[0], max_nb_variables))
y_test = labels[(np.array(test_ind) - 1)]

max_variables_test = -np.inf
count = 0

for i in range(X.shape[0]):
    var_count = X[i].shape[-1]

    if var_count > max_nb_variables:
        max_variables_test = var_count
        count += 1

print('max nb variables test : ', max_variables_test)
print("# of instances where test vars > %d : " % max_nb_variables, count)

print("\nSince there is only %d instance where test # variables > %d (max # of variables in train), "
      "we clip the specific instance to match %d variables\n" % (max_nb_variables, count, max_nb_variables))

# pad ending with zeros to get numpy arrays
for i in range(X_test.shape[0]):
    var_count = X[i].shape[-1]
    X_test[i, :, :var_count] = X[i][:, :max_nb_variables]

''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Nb classes : ", len(np.unique(y_train)))

np.save(action_3d_path + 'X_train.npy', X_train)
np.save(action_3d_path + 'y_train.npy', y_train)
np.save(action_3d_path + 'X_test.npy', X_test)
np.save(action_3d_path + 'y_test.npy', y_test)