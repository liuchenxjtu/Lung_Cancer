from keras.layers import Flatten, Input
from keras.layers import AveragePooling3D, MaxPooling3D

from keras.models import Model

from keras import backend as K

import numpy as np
import pandas as pd

from sklearn.externals import joblib


def generate_spatial_agg_features(X, input_shape=(11, 11, 11, 256)):
    img_input = Input(shape=input_shape)

    x = MaxPooling3D((3, 3, 3), strides=(3, 3, 3), name='block1_pool', padding='same')(img_input)
    # x = AveragePooling3D((3, 3, 3), strides=(3, 3, 3), name='block1_pool', padding='same')(img_input)
    x = Flatten(name='flatten')(x)

    model = Model(inputs=img_input, outputs=x)

    return model.predict(X)

K.set_image_data_format('channels_last')

df_train = pd.read_csv('data/stage1_labels.csv')
df_val = pd.read_csv('data/stage1_solution.csv')

train_chunks = df_train.id.apply(lambda x: np.load('feature/%s.npy' % str(x)))

train_chunks = np.dstack(train_chunks)
train_chunks = np.rollaxis(train_chunks, -1).reshape(train_chunks.shape[2], 11, 11, 11, 256)
X_train = generate_spatial_agg_features(train_chunks)
y_train = df_train.cancer.astype(int)

joblib.dump(X_train, 'data/X_train.pkl')
joblib.dump(y_train, 'data/y_train.pkl')

val_chunks = df_val.id.apply(lambda x: np.load('feature/%s.npy' % str(x)))

val_chunks = np.dstack(val_chunks)
val_chunks = np.rollaxis(val_chunks, -1).reshape(val_chunks.shape[2], 11, 11, 11, 256)
X_val = generate_spatial_agg_features(val_chunks)
y_val = df_val.cancer.astype(int)

joblib.dump(X_val, 'data/X_val.pkl')
joblib.dump(y_val, 'data/y_val.pkl')












