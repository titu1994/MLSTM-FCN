import scipy.io as sio

import numpy as np

ck_path = r''
DATA = sio.loadmat(ck_path + 'randomperm_CK.mat')

var_list = None
no_folds = 10

for k in range(10):
    index_fold = k + 1 # index fold
    pre = 0

    train_ind = []
    test_ind = []
    each_label_number = DATA['each_label_number'][0]
    labels_n = DATA['labels_n'][0]

    for i in range(0, labels_n[0]):
        fold_size = np.floor(each_label_number[i] / no_folds)
        ind = (index_fold - 1) * fold_size + 1
        train_ind.append(list(range(pre + 1, int(pre + ind - 1 + 1))))
        train_ind.append(list(range(int(pre + ind + fold_size), int(pre + 1 + each_label_number[i]))))
        test_ind.append(list(range(int(pre + ind), int(pre + ind + fold_size - 1 + 1))))
        pre = pre + each_label_number[i]

    trainind = [item for sublist in train_ind for item in sublist]
    testind = [item for sublist in test_ind for item in sublist]

    X = DATA['new_X'][0, (np.array(trainind) - 1)]  ###Train data
    y = DATA['new_labels'][0, (np.array(trainind) - 1)]  ###Train Labels

    ''' Load train Dataset '''
    if var_list is None:
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

        print('-' * 80)
        print()

    X_train = np.zeros((X.shape[0], X[0].shape[0], max_nb_timesteps))
    y_train = y

    # pad ending with zeros to get numpy arrays
    for i in range(X_train.shape[0]):
        var_count = X[i].shape[-1]
        X_train[i, :, :var_count] = X[i]

    ''' Load test dataset'''
    X = DATA['new_X'][0, (np.array(testind) - 1)]  ####Test Data
    y = DATA['new_labels'][0, (np.array(testind) - 1)]  ####Test Labels

    X_test = np.zeros((X.shape[0], X[0].shape[0], max_nb_timesteps))
    y_test = y

    # pad ending with zeros to get numpy arrays
    for i in range(X_test.shape[0]):
        var_count = X[i].shape[-1]
        X_test[i, :, :var_count] = X[i]

    ''' Save the datasets '''
    print("Dataset Fold #%d" % index_fold)
    print("Train dataset : ", X_train.shape, y_train.shape)
    print("Test dataset : ", X_test.shape, y_test.shape)
    print("Train dataset metrics : ", X_train.mean(), X_train.std())
    print("Test dataset : ", X_test.mean(), X_test.std())
    print("Nb classes : ", len(np.unique(y_train)), "Classes : ", np.unique(y_train))

    np.save(ck_path + 'X_train_%d.npy' % index_fold, X_train)
    np.save(ck_path + 'y_train_%d.npy' % index_fold, y_train)
    np.save(ck_path + 'X_test_%d.npy' % index_fold, X_test)
    np.save(ck_path + 'y_test_%d.npy' % index_fold, y_test)

    print()
