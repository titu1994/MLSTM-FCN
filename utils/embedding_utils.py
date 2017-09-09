import os
import pickle
import numpy as np
np.random.seed(1000)

from utils.generic_utils import load_dataset_at

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_embedding_matrix(embedding_path, word_index, max_nb_words, embedding_dim, print_error_words=True):
    if not os.path.exists('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words, embedding_dim)):
        embeddings_index = {}
        error_words = []

        f = open(embedding_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                error_words.append(word)

        f.close()

        if len(error_words) > 0:
            print("%d words were not added." % (len(error_words)))
            if print_error_words:
                print("Words are : \n", error_words)

        print('Preparing embedding matrix.')

        # prepare embedding matrix
        nb_words = min(max_nb_words, len(word_index))
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        np.save('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words,
                                                                             embedding_dim),
                embedding_matrix)

        print('Saved embedding matrix')

    else:
        embedding_matrix = np.load('data/embedding_matrix max words %d embedding dim %d.npy' % (max_nb_words,
                                                                                                embedding_dim))

        print('Loaded embedding matrix')

    return embedding_matrix


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def prepare_tokenized_data(texts, max_nb_words, max_sequence_length, ngram_range=1):
    if not os.path.exists('data/tokenizer.pkl'):
        tokenizer = Tokenizer(nb_words=max_nb_words)
        tokenizer.fit_on_texts(texts)

        with open('data/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        print('Saved tokenizer.pkl')
    else:
        with open('data/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print('Loaded tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique 1-gram tokens.' % len(word_index))

    ngram_set = set()
    for input_list in sequences:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_nb_words + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}
    word_index.update(token_indice)

    max_features = np.max(list(indice_token.keys())) + 1
    print('Now there are:', max_features, 'features')

    # Augmenting X_train and X_test_mat with n-grams features
    sequences = add_ngram(sequences, token_indice, ngram_range)
    print('Average sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int)))
    print('Max sequence length: {}'.format(np.max(list(map(len, sequences)))))

    data = pad_sequences(sequences, maxlen=max_sequence_length)


    return (data, word_index)


# def __load_embeddings(dataset_prefix, verbose=False):
#
#     embedding_path = npy_path # change to numpy data format (which contains the preloaded embedding matrix)
#     if os.path.exists(embedding_path):
#         # embedding matrix exists, no need to create again.
#         print("Loading embedding matrix for dataset \'%s\'" % (dataset_prefix))
#         embedding_matrix = np.load(embedding_path)
#         return embedding_matrix
#
#     with open(txt_path, 'r', encoding='utf8') as f:
#         header = f.readline()
#         splits = header.split(' ')
#
#         vocab_size = int(splits[0])
#         embedding_size = int(splits[1])
#
#         embeddings_index = {}
#         error_words = []
#
#         for line in f:
#             values = line.split()
#             word = values[0]
#             try:
#                 coefs = np.asarray(values[1:], dtype='float32')
#                 embeddings_index[word] = coefs
#             except Exception:
#                 error_words.append(word)
#
#         if len(error_words) > 0:
#             print("%d words were not added." % (len(error_words)))
#             if verbose:
#                 print("Words are : \n", error_words)
#
#         if verbose: print('Preparing embedding matrix.')
#
#         embedding_matrix = np.zeros((vocab_size, embedding_size))
#
#         for key, vector in embeddings_index.items():
#             if vector is not None:
#                 # words not found in embedding index will be all-zeros.
#                 key = int(key)
#                 embedding_matrix[key] = vector
#
#         if verbose: print('Saving embedding matrix for dataset \'%s\'' % (dataset_prefix))
#
#         np.save(embedding_path, embedding_matrix)
#         return embedding_matrix
