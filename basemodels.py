import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization 
from tensorflow.keras.layers import  Activation, Dropout, TimeDistributed
from tensorflow.keras.layers import LSTM, RepeatVector, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import anomalyutils

def set_random(seed_value):
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

set_random(0)

## AE
def autoencoder(data):
    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("in autoencoder\n")

    input_dots = Input((data.shape[1],))

    x = Dense(5)(input_dots)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    bottleneck = Dense(2, activation='linear')(x)

    x = Dense(4)(bottleneck)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(5)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    out = Dense(data.shape[1], activation='linear')(x)

    model = Model(input_dots, out)
    model.compile(optimizer=Adam(0.005), loss='mae', metrics=["mse"])
    
    early_stopping = EarlyStopping(patience=3, verbose=0)
    model.fit(data, data,
                validation_split=0.2,
                epochs=40,
                batch_size=32,
                verbose=0,
                shuffle=True,
                callbacks=[early_stopping]
            )
    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("Finished autoencoder\n")
    return model

## Conv_AE
def conv_ae(data):
    model = keras.Sequential(
        [
            layers.Input(shape=(data.shape[1], data.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    # model.summary()

    history = model.fit(
        data,
        data,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
        ],
    )
    return model#, history
    

## LSTM
def lstm(data_x, data_y, name=None):
    name_of_method = ""
    if name is None:
        name_of_method = "lstm.h5"
    else:
        name_of_method = f"lstm_{name}.h5"
    
    n_features = data_x.shape[2]
    
    # model defining
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(5, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mae', metrics=["mse"])
    
    # callbacks defining
    early_stopping = EarlyStopping(patience=10, verbose=0)
    model_checkpoint = ModelCheckpoint(name_of_method, save_best_only=True, verbose=0, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=0)
    
    # model fitting
    history = model.fit(data_x, data_y,
                        validation_split=0.2,
                        epochs=25,
                        batch_size=32,
                        verbose=0,
                        shuffle=False,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr])
    return model

## LSTM_AE
def lstm_ae(data):
    # model defining
    # define encoder
    inputs = keras.Input(shape=(data.shape[1], data.shape[2]))
    encoded = layers.LSTM(100, activation='relu')(inputs)

    # define reconstruct decoder
    decoded = layers.RepeatVector(data.shape[1])(encoded)
    decoded = layers.LSTM(100, activation='relu', return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(data.shape[2]))(decoded)

    # tie it together
    model = keras.Model(inputs, decoded)
    encoder = keras.Model(inputs, encoded)

    model.compile(optimizer='adam', loss='mae', metrics=["mse"])
    
    # callbacks defining
    early_stopping = EarlyStopping(patience=5, verbose=0)
#     reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=0)
    
    # model fitting
    history = model.fit(data, data,
                        validation_split=0.1,
                        epochs=100,
                        batch_size=32,
                        verbose=0,
                        shuffle=False,
                        callbacks=[early_stopping]#, reduce_lr]
                        )
    return model

## LSMTM_VAE
def create_lstm_vae(input_dim, 
    timesteps, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    epsilon_std):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.
    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    vae.compile(optimizer='rmsprop', loss="mae")
    
    return vae, encoder, generator

def lstm_vae(data):
    input_dim = data.shape[-1] # 13
    timesteps = data.shape[1] # 3
    BATCH_SIZE = 1
    
    model, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=BATCH_SIZE, 
        intermediate_dim=32,
        latent_dim=100,
        epsilon_std=1.)

    history = model.fit(
        data,
        data,
        epochs=20,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
        ],
    )
    return model


def multi_model(df):
    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("in multi_model\n")
    X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)

    # scaler init and fitting
    StSc = StandardScaler()
    StSc.fit(X_train)

    X_ae = StSc.transform(X_train)

    # model defining and fitting
    autoencoder_model = autoencoder(X_ae)

    predictions_ae = anomalyutils.get_ae_predicts(autoencoder_model, X_ae)
    residuals_autoencoder = anomalyutils.get_ae_residuals(X_ae, predictions_ae)
    UCL_autoencoder = residuals_autoencoder.quantile(0.99)

    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("Finished multi_model\n")
    return UCL_autoencoder

def double(x):
    return x