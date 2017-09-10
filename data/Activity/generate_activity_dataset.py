import scipy.io as sio
import numpy as np


def find1(a, func):
    qqq = [i for (i, val) in enumerate(a) if func(val)]
    return (np.array(qqq) + 1)


activity_path = r""

DATA = sio.loadmat(activity_path + 'joint3D_feature_noFFT.mat')

Joint3D_feature = DATA['Joint3D_feature'][0]
labels = DATA['labels'][0]

data = Joint3D_feature
K = 16
train_ind = []
test_ind = []
testActors = [1, 2, 3, 4, 5]
testClass = range(1, 17)
true_i = 0

for a in range(1, 17):
    for j in range(1, 11):
        for e in range(1, 3):
            true_i = true_i + 1
            if not (np.all((find1(testActors, lambda x: x == j)) == 0)):
                test_ind.append(true_i)
            else:
                train_ind.append(true_i)

''' Load train set '''
X = data[(np.array(train_ind) - 1)]

var_list = []
for i in range(X.shape[0]):
    var_count = X[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
max_nb_timesteps = var_list.max()
min_nb_timesteps = var_list.min()
median_nb_timesteps = np.median(var_list)

print('max nb timesteps train : ', max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)


X_train = np.zeros((X.shape[0], X[0].shape[0], max_nb_timesteps))
y_train = labels[(np.array(train_ind) - 1)]

# pad ending with zeros to get numpy arrays
for i in range(X_train.shape[0]):
    var_count = X[i].shape[-1]
    X_train[i, :, :var_count] = X[i]

''' Load test set '''
X = data[(np.array(test_ind) - 1)]

X_test = np.zeros((X.shape[0], X[0].shape[0], max_nb_timesteps))
y_test = labels[(np.array(test_ind) - 1)]

max_variables_test = -np.inf
count = 0

for i in range(X.shape[0]):
    var_count = X[i].shape[-1]

    if var_count > max_nb_timesteps:
        max_variables_test = var_count
        count += 1

print('max nb variables test : ', max_variables_test)
print("# of instances where test vars > %d : " % max_nb_timesteps, count)

print("\nSince there is only %d instance where test # variables > %d (max # of variables in train), "
      "we clip the specific instance to match %d variables\n" % (count, max_nb_timesteps, max_nb_timesteps))

# pad ending with zeros to get numpy arrays
for i in range(X_test.shape[0]):
    var_count = X[i].shape[-1]
    X_test[i, :, :var_count] = X[i][:, :max_nb_timesteps]

''' Save the datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

np.save(activity_path + 'X_train.npy', X_train)
np.save(activity_path + 'y_train.npy', y_train)
np.save(activity_path + 'X_test.npy', X_test)
np.save(activity_path + 'y_test.npy', y_test)
