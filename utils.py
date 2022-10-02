import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import roc_auc_score, f1_score
import itertools

def set_random(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

set_random(0)

def get_files_and_names():
    # benchmark files checking
    all_files=[]
    all_files_name = []
    for root, dirs, files in os.walk("./data/"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
                all_files_name.append(root.split("/")[-1] + "_" + file.replace(".", "_"))
    return all_files, all_files_name


def get_anomaly_data_and_names():
    all_files, all_files_name = get_files_and_names()
    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file, 
                            sep=';', 
                            index_col='datetime', 
                            parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    list_of_names = [file for file in all_files_name if 'anomaly-free' not in file]
    return list_of_df, list_of_names

def get_anomaly_free_data_and_names():
    all_files, all_files_name = get_files_and_names()
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], 
                                sep=';', 
                                index_col='datetime', 
                                parse_dates=True)
    list_of_names = [file for file in all_files_name if 'anomaly-free' in file]
    return anomaly_free_df, list_of_names

def get_files():
    # benchmark files checking
    all_files=[]
    for root, dirs, files in os.walk("./data/"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    return all_files


def get_anomaly_data():
    all_files = get_files()
    # datasets with anomalies loading
    list_of_df = [pd.read_csv(file, 
                            sep=';', 
                            index_col='datetime', 
                            parse_dates=True) for file in all_files if 'anomaly-free' not in file]

    return list_of_df

def get_anomaly_free_data():
    all_files = get_files()
    anomaly_free_df = pd.read_csv([file for file in all_files if 'anomaly-free' in file][0], 
                                sep=';', 
                                index_col='datetime', 
                                parse_dates=True)
    return anomaly_free_df

def create_sequences(values, time_steps):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_anomalies_labels_from_continuous_windows(original_X_data, residuals, N_STEPS, UCL, df_index):
    anomalous_data = residuals > (3/2 * UCL)
    anomalous_data_indices = []
    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)
    
    prediction = pd.Series(data=0, index=df_index)
    prediction.iloc[anomalous_data_indices] = 1
    return prediction

def get_scores_from_residuals(original_X_data, residuals, N_STEPS, df_index):

    prediction = pd.Series(data=0, index=df_index)

    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        score = residuals[data_idx]
        prediction.iloc[data_idx] = score

    return prediction

def get_scores_scoring_with_current_window(original_X_data, residuals, N_STEPS, df_index):
    
    prediction = pd.Series(data=0, index=df_index)
    
    for data_idx in range(N_STEPS - 1, len(original_X_data)):
        score = residuals[data_idx]
        prediction.iloc[data_idx] = score

    return prediction

def get_scores_scoring_with_mean_steps_windows(original_X_data, residuals, N_STEPS, df_index):
    
    prediction = pd.Series(data=0, index=df_index)

    for data_idx in range(N_STEPS - 1, len(original_X_data) - N_STEPS + 1):
        score = np.mean(residuals[data_idx - N_STEPS + 1 : data_idx])
        prediction.iloc[data_idx] = score
    
    return prediction

def plot_single_confusion_matrix(df, y_actual, y_predicted):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure()
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)


    confusion_matrix = pd.crosstab(df[y_actual], df[y_predicted], rownames=['Actual'], colnames=['Predicted'])
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='g')
    
    plt.title(y_predicted.replace("anomaly_by_",""))
    plt.show()

def get_x_y(n):
   x = -1
   y = -1
   if int(math.sqrt(n)) == math.sqrt(n):
      x = int(math.sqrt(n))+1
      y = int(math.sqrt(n))+1
      return x, y
   
   m = n
   while int(math.sqrt(m)) == math.sqrt(m):
      m-=1

   x = int(math.sqrt(m))+1
   y = int(math.sqrt(m))+1
   return x, y

def plot_confusion_matrix_all(df, y_actual, *cols, save_image=None):
    px = 1/plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1920*px, 1080*px))
    # fig = plt.figure(figsize = (24,8))
    fig.patch.set_facecolor((1,1,1,1))
    fig.patch.set_alpha(1.0)

    x, y = get_x_y(len(cols))


    fig.subplots_adjust(hspace=0.8, wspace=0.5)

    for i,col in enumerate(cols):
        ax = fig.add_subplot(x, y, i+1)

        roc_number = roc_auc_score(df[y_actual], df[col])

        model_title = col.replace("anomaly_by_","")
        F1 = f1_score(df[y_actual], df[col])
        custom_title = f"{model_title}\nAUC: {roc_number}\nF1 Score: {F1}"
        confusion_matrix = pd.crosstab(df[y_actual], df[col], rownames=['Actual'], colnames=['Predicted'])
        g = sns.heatmap(confusion_matrix, annot=True, fmt='g', ax = ax).set_title(custom_title)

    if save_image is not None:
        image_path = str(save_image)
        # remove white space
        plt.savefig(image_path, bbox_inches='tight')
    fig.tight_layout()
    plt.show()


def get_bagg_features(T=-1):
    seq = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
    comb_size_4 = list(itertools.combinations(seq,4))
    comb_size_5 = list(itertools.combinations(seq,5))
    comb_size_6 = list(itertools.combinations(seq,6))
    comb_size_7 = list(itertools.combinations(seq,7))
    size_N = 0
    if ((T<=0) or (T>162)):
        return [seq]
    elif ((T>0) or (T<=162)):
        all_lists = []
        for i in range(0,T):
            choose_comb = []
            if len(comb_size_4)!=0:
                choose_comb.append(4)
            if len(comb_size_5)!=0:
                choose_comb.append(5)
            if len(comb_size_6)!=0:
                choose_comb.append(6)
            if len(comb_size_7)!=0:
                choose_comb.append(7)
            
            size_N = random.choice(choose_comb)
            
            tmp_list_ind = None
            if size_N==4:
                tmp_list_ind = random.randint(0, len(comb_size_4)-1)
                all_lists.append(comb_size_4.pop(tmp_list_ind))
            if size_N==5:
                tmp_list_ind = random.randint(0, len(comb_size_5)-1)
                all_lists.append(comb_size_5.pop(tmp_list_ind))
            if size_N==6:
                tmp_list_ind = random.randint(0, len(comb_size_6)-1)
                all_lists.append(comb_size_6.pop(tmp_list_ind))
            if size_N==7:
                tmp_list_ind = random.randint(0, len(comb_size_7)-1)
                all_lists.append(comb_size_7.pop(tmp_list_ind))
        return all_lists
    else:
        return None