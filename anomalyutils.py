
import pandas as pd
import numpy as np

def set_random(seed_value):
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

set_random(0)


def get_ae_residuals(data_true, data_predicted):
    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("in get_ae_residuals")

    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("Finished get_ae_residuals\n")
    return pd.DataFrame(data_true - data_predicted).abs().sum(axis=1)

def get_conv_ae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))

def get_lstm_residuals(data_true, data_predicted):
    return pd.DataFrame(data_true - data_predicted).abs().sum(axis=1)

def get_lstm_ae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))

def get_lstm_vae_residuals(data_true, data_predicted):
    return pd.Series(np.sum(np.mean(np.abs(data_true - data_predicted), axis=1), axis=1))


def get_ae_predicts(model, data):
    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("in get_ae_predicts\n")

    with open("info_pool.txt", "a") as file_object:
        # Append 'hello' at the end of file
        file_object.write("Finished get_ae_predicts\n")
    return model.predict(data)

def get_conv_ae_predicts(model, data):
    return model.predict(data)

def get_lstm_predicts(model, data):
    return model.predict(data)

def get_lstm_ae_predicts(model, data):
    return model.predict(data)

def get_lstm_vae_predicts(model, data):
    return model.predict(data)