import scipy.io as sio

import numpy as np

ck_path = r'../data/CK/'
DATA = sio.loadmat(ck_path + 'randomperm_CK.mat')

max_sequence_length = None
min_sequence_length = None
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
    if max_sequence_length is None:
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
        print('-' * 80)
        print()

    X_train = np.zeros((X.shape[0], X[0].shape[0], max_sequence_length))
    y_train = y

    # pad ending with zeros to get numpy arrays
    for i in range(X_train.shape[0]):
        var_count = X[i].shape[-1]
        X_train[i, :, :var_count] = X[i]

    ''' Load test dataset'''
    X = DATA['new_X'][0, (np.array(testind) - 1)]  ####Test Data
    y = DATA['new_labels'][0, (np.array(testind) - 1)]  ####Test Labels

    X_test = np.zeros((X.shape[0], X[0].shape[0], max_sequence_length))
    y_test = y

    # pad ending with zeros to get numpy arrays
    for i in range(X_test.shape[0]):
        var_count = X[i].shape[-1]
        X_test[i, :, :var_count] = X[i]

    ''' Save the datasets '''
    print("Dataset Fold #%d" % index_fold)
    print("Train dataset : ", X_train.shape, y_train.shape)
    print("Test dataset : ", X_test.shape, y_test.shape)
    print("Nb classes : ", len(np.unique(y_train)), "Classes : ", np.unique(y_train))

    np.save(ck_path + 'X_train_%d.npy' % index_fold, X_train)
    np.save(ck_path + 'y_train_%d.npy' % index_fold, y_train)
    np.save(ck_path + 'X_test_%d.npy' % index_fold, X_test)
    np.save(ck_path + 'y_test_%d.npy' % index_fold, y_test)

    print()
