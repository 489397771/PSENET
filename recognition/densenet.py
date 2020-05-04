from keras.layers.core import Dense, Dropout, Activation, Permute
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, Flatten, Bidirectional, MaxPooling2D, LSTM
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras import Model
from . import keys


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def dense_black(x, nb_layers, nb_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    if pooltype == 2:
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif pooltype == 1:
        x = ZeroPadding2D(padding=(0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif pooltype == 3:
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)

    return x, nb_filter


def dense_cnn(input, nclass):
    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64

    # conv 64 5 * 5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    # 64 + 8 * 8 = 128
    x, _nb_filter = dense_black(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_black(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 3, _weight_decay)
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_black(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=1, epsilon=1.1e-5)(x)
    x = Conv2D(512, kernel_size=(2, 2), strides=(2, 1), activation='relu', padding='valid', name='conv7')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', name='pool4')(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)

    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred


def dense_blstm(input):
    pass


rnnunit = 256


def bcrnn(input, nclass):
    m = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid', name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid', name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)

    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='flatten')(m)

    m = Bidirectional(LSTM(rnnunit, return_sequences=True), name='blstm1-1')(m)
    m = Dense(rnnunit, name='blstm1_out-1', activation='linear')(m)
    m = Bidirectional(LSTM(rnnunit, return_sequences=True), name='blstm2-1')(m)
    y_pred = Dense(nclass, name='blstm2_out-2', activation='softmax')(m)

    return y_pred


if __name__ == '__main__':
    character = keys.alphabetChinese[:]
    character = character[1:] + u'卍'
    nclass = len(character)
    print(nclass)
    input = Input(shape=(32, None, 1), name='the_input')
    y_pred = dense_cnn(input=input, nclass=nclass)
    model = Model(input=input, outputs=y_pred)
    json_string = model.to_json()
    print(type(json_string))
    with open('weights2.json', 'w')as f:
        f.write(json_string)
