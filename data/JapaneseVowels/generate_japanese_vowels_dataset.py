import numpy as np
from sklearn.model_selection import KFold

japanese_vowels_path = ''
train_data_path = japanese_vowels_path + 'ae.train'
test_data_path = japanese_vowels_path + 'ae.test'
train_speaker_rows_path = japanese_vowels_path + 'size_ae.train'
test_speaker_rows_path = japanese_vowels_path + 'size_ae.test'

''' TRAIN '''

train_arrays = []
with open(train_data_path, 'r') as f:
    array_buffer = []

    for line in f:
        if line == '\n':
            train_arrays.append(array_buffer)
            array_buffer = []
        else:
            array = np.array(line.split(' ')[:-1], dtype='float32')
            array_buffer.append(array)

train_array_lengths = []
for i in range(len(train_arrays)):
    train_array_lengths.append(len(train_arrays[i]))

train_array_lengths = np.asarray(train_array_lengths)
max_train_length = train_array_lengths.max()
nb_variables = len(train_arrays[0][0])

max_nb_timesteps = train_array_lengths.max()
min_nb_timesteps = train_array_lengths.min()
median_nb_timesteps = np.median(train_array_lengths)

print('max nb timesteps train : ', max_nb_timesteps)
print('min nb timesteps train : ', min_nb_timesteps)
print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)

X_train = np.zeros((len(train_arrays), nb_variables, max_train_length))
y_train = np.zeros((len(train_arrays),))

for i in range(len(train_arrays)):
    x = np.asarray(train_arrays[i])
    x = x.transpose((1, 0))
    max_len = x.shape[-1]
    X_train[i, :, :max_len] = x

with open(train_speaker_rows_path, 'r') as f:
    lengths = f.readline().split(' ')
    try:
        test = int(lengths[-1])
    except ValueError:
        lengths = lengths[:-1]
    lengths = np.array(lengths, dtype='int')

    index = 0
    label = 0
    for length in lengths:
        for _ in range(length):
            y_train[index] = label
            index += 1
        label += 1

''' TEST '''

test_arrays = []
with open(test_data_path, 'r') as f:
    array_buffer = []

    for line in f:
        if line == '\n':
            test_arrays.append(array_buffer)
            array_buffer = []
        else:
            array = np.array(line.split(' ')[:-1], dtype='float32')
            array_buffer.append(array)

X_test = np.zeros((len(test_arrays), nb_variables, max_train_length))
y_test = np.zeros((len(test_arrays),))

for i in range(len(test_arrays)):
    x = np.asarray(test_arrays[i])
    x = x.transpose((1, 0))
    max_len = x.shape[-1]
    X_test[i, :, :max_len] = x[:, :max_train_length]

with open(test_speaker_rows_path, 'r') as f:
    lengths = f.readline().split(' ')
    try:
        test = int(lengths[-1])
    except ValueError:
        lengths = lengths[:-1]
    lengths = np.array(lengths, dtype='int')

    index = 0
    label = 0
    for length in lengths:
        for _ in range(length):
            y_test[index] = label
            index += 1
        label += 1

''' Save datasets '''
print("Train dataset : ", X_train.shape, y_train.shape)
print("Test dataset : ", X_test.shape, y_test.shape)
print("Train dataset metrics : ", X_train.mean(), X_train.std())
print("Test dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

#print("\nPerforming 10 fold crossvalidation split now")
#kf = KFold(n_splits=10, shuffle=False, random_state=1000)

np.save(japanese_vowels_path + 'X_train_mat.npy', X_train)
np.save(japanese_vowels_path + 'y_train_mat.npy', y_train)
np.save(japanese_vowels_path + 'X_test_mat.npy', X_test) # full test dataset
np.save(japanese_vowels_path + 'y_test_mat.npy', y_test) # full test dataset

np.save(japanese_vowels_path + 'X_train.npy', X_train)
np.save(japanese_vowels_path + 'y_train.npy', y_train)
np.save(japanese_vowels_path + 'X_test.npy', X_test) # full test dataset
np.save(japanese_vowels_path + 'y_test.npy', y_test) # full test dataset