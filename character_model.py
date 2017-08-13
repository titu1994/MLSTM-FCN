from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation, Masking
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 2

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True


def generate_model():
    ip = Input(shape=(MAX_TIMESTEPS, MAX_NB_VARIABLES))

    x = Masking()(ip)
    x = LSTM(128)(x)
    x = Dropout(0.8)(x)

    #y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


def generate_model_2():
    ip = Input(shape=(MAX_TIMESTEPS, MAX_NB_VARIABLES))

    x = Masking()(ip)
    x = AttentionLSTM(128)(x)
    x = Dropout(0.8)(x)

    #y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == "__main__":
    from keras import backend as K
    import json

    ''' Train portion '''
    scores = []

    for i in range(10):
        K.clear_session()

        print("Begin iteration %d" % (i + 1))
        print("*" * 80)
        print()

        model = generate_model() # change to generate_model_2()
        train_model(model, DATASET_INDEX, dataset_prefix='character', dataset_fold_id=(i + 1), epochs=600, batch_size=128)
        score = evaluate_model(model, DATASET_INDEX, dataset_prefix='character', dataset_fold_id=(i + 1), batch_size=128)
        scores.append(score)

    with open('data/character/scores.json', 'w') as f:
        json.dump({'scores': scores}, f)

    ''' evaluate average score '''
    with open('data/character/scores.json', 'r') as f:
        results = json.load(f)

    scores = results['scores']
    avg_score = sum(scores) / len(scores)
    print("Scores : ", scores)
    print("Average score over 10 epochs : ", avg_score)

    #visualize_context_vector(model, DATASET_INDEX, dataset_prefix='ck',
    #                         visualize_sequence=True, visualize_classwise=True, limit=1)

    # visualize_cam(model, DATASET_INDEX, dataset_prefix='ck', class_id=0)
