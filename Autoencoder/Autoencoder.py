from MultiProcess import load_data
from build_model import AE
import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Reshape, Dropout
import os

# load data
print("Load data...")
train, train_label, test, test_label, true_train_label, true_test_label = load_data()  # data:(, 196) label:(, 10)
print("train shape: ", train.shape)
train_label = train_label.reshape((-1, 10))
test_label = test_label.reshape((-1, 10))
print("train_label shape: ", train_label.shape)


# build model
print("Build AE model")
autoencoder_1, encoder_1, autoencoder_2, encoder_2, autoencoder_3, encoder_3, sSAE, SAE_encoder = AE()

autoencoder_1.summary()
encoder_1.summary()
autoencoder_2.summary()
encoder_2.summary()
autoencoder_3.summary()
encoder_3
sSAE.summary()
SAE_encoder.summary()


print("Start pre-training....")

# fit the first layer, 在此处添加validation_data=test，加上callbacks，记录的是val_loss，取最小的那个
print("First layer training....")
AE_1_dir = os.path.join(os.getcwd(), 'AE')
ae_1_filepath="best_ae_1.hdf5"
ae_1_point = ModelCheckpoint(os.path.join(AE_1_dir, ae_1_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_1_stops = EarlyStopping(monitor='val_loss', patience=10, mode='min')
autoencoder_1.fit(train, train, epochs=100, batch_size=2048, validation_data=(test, test), verbose=0, shuffle=True, callbacks=[ae_1_point, ae_1_stops])

autoencoder_1.load_weights('./AE/best_ae_1.hdf5')
first_layer_output = encoder_1.predict(train)  # 在此使用loss最小的那个模型
test_first_out = encoder_1.predict(test)
print("The shape of first layer output is: ", first_layer_output.shape)


# fit the second layer
print("Second layer training....")
AE_2_dir = os.path.join(os.getcwd(), 'AE')
ae_2_filepath="best_ae_2.hdf5"
ae_2_point = ModelCheckpoint(os.path.join(AE_2_dir, ae_2_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_2_stops = EarlyStopping(monitor='val_loss', patience=8, mode='min')
autoencoder_2.fit(first_layer_output, first_layer_output, epochs=100, batch_size=2048, verbose=0, validation_data=(test_first_out, test_first_out), shuffle=True, callbacks=[ae_2_point, ae_2_stops])

autoencoder_2.load_weights('./AE/best_ae_2.hdf5')
second_layer_output = encoder_2.predict(first_layer_output)
test_second_out = encoder_2.predict(test_first_out)
print("The shape of second layer output is: ", second_layer_output.shape)



# fit the third layer
print("Third layer training....")
AE_3_dir = os.path.join(os.getcwd(), 'AE')
ae_3_filepath="best_ae_3.hdf5"
ae_3_point = ModelCheckpoint(os.path.join(AE_3_dir, ae_3_filepath), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
ae_3_stops = EarlyStopping(monitor='val_loss', patience=8, mode='min')
autoencoder_3.fit(second_layer_output, second_layer_output, epochs=150, batch_size=2048, verbose=0, validation_data=(test_second_out, test_second_out), shuffle=True, callbacks=[ae_3_point, ae_3_stops])
autoencoder_3.load_weights('./AE/best_ae_3.hdf5')



print("Pass the weights to SAE_encoder...")
SAE_encoder.layers[1].set_weights(autoencoder_1.layers[1].get_weights())  # first Dense
SAE_encoder.layers[2].set_weights(autoencoder_1.layers[2].get_weights())  # first BN
SAE_encoder.layers[3].set_weights(autoencoder_2.layers[1].get_weights())  # second Dense
SAE_encoder.layers[4].set_weights(autoencoder_2.layers[2].get_weights())  # second BN
SAE_encoder.layers[5].set_weights(autoencoder_3.layers[1].get_weights())  # third Dense
SAE_encoder.layers[6].set_weights(autoencoder_3.layers[2].get_weights())  # third BN
encoded_train = SAE_encoder.predict(train)
encoded_test = SAE_encoder.predict(test)

import scipy.io as io
io.savemat('./autoencoder', {'ae_train':encoded_train})
io.savemat('./autoencoder', {'ae_test':encoded_test})