from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, Input
from utils.keras_sparisity_regularization import SparsityRegularization
from keras.models import Model


def vgg(nb_classes=10, sparse_factor=0.0001, prune_rate=0):
    model_inputs = Input(shape=(32, 32, 3))
    keep_rate = 1 - prune_rate
    x = Conv2D(int(32 * keep_rate), (3, 3), padding='same', activation='relu')(model_inputs)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = Conv2D(int(32 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(int(64 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = Conv2D(int(64 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(int(128 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = Conv2D(int(128 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = Conv2D(int(128 * keep_rate), (3, 3), padding='same', activation='relu')(x)
    x = SparsityRegularization(l1=sparse_factor)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=model_inputs, outputs=x)
    return model
