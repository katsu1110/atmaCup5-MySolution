"""
get CNN features

:USAGE:
>>> python add_cnn_features.py

"""

### Tricks ###
FLIP = False

### Libraries ###
import numpy as np
import pandas as pd
import os, sys
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import layers

### data location ###
input_path = "../input/"
output_path = "../output/"

### load data ###
def read_data():
    # load
    train = pd.read_csv(input_path + "train.csv")
    y_train = train['target'].values
    submission = pd.read_csv(input_path + "atmaCup5__sample_submission.csv")

    # waves
    tr_waves = np.load(output_path + "tr_wave.npy")
    ts_waves = np.load(output_path + "ts_wave.npy")

    # sample-wise normalize
    for i in range(tr_waves.shape[0]):
        tr_waves[i, :] = tr_waves[i, :] / np.std(tr_waves[i, :])
    
    for i in range(ts_waves.shape[0]):
        ts_waves[i, :] = ts_waves[i, :] / np.std(ts_waves[i, :])

    return submission, tr_waves, ts_waves, y_train
sub, X_train, X_test, y_train = read_data()
orig_trlen = X_train.shape[0]

## augmentation
if FLIP:
    # flip positive label
    aug_waves = X_train[y_train == 1, :]
    X_train = np.vstack((X_train, np.fliplr(aug_waves)))
    y_train = np.hstack((y_train, np.ones(aug_waves.shape[0])))
    print("FLIP : train len {:,} to {:,}".format(orig_trlen, X_train.shape[0]))

## crop around the peak
L = 400
def get_peakwave(wave):
    new_wave = np.zeros((wave.shape[0], L))
    for i in range(wave.shape[0]):
        peak_idx = np.argmax(wave[i, :])
        if peak_idx < L:
            new_wave[i, :] = wave[i, :L]
        elif peak_idx > 511 - L:
            new_wave[i, :] = wave[i, -L:]
        else:
            new_wave[i, :] = wave[i, peak_idx - L // 2 : peak_idx + L // 2]
    new_wave = np.reshape(new_wave, (-1, L, 1))
    return new_wave
X_train = get_peakwave(X_train)
X_test = get_peakwave(X_test)
y_train = y_train.reshape(-1, 1)

### CNN modeling ###
## for CNN features
def extra_dist(best_model, X_skf_train, X_skf_test):

    ## 中間層を出力するためのモデル
    layer_name = 'hidden'
    hidden_model = models.Model(inputs=best_model.input,
                         outputs=best_model.get_layer(layer_name).output)


    ## trainを入力し、64次元のベクトル抽出
    hidden_pred_train = hidden_model.predict(X_skf_train)
    hidden_pred_test = hidden_model.predict(X_skf_test)
    
    ## PCA
    trans = decomposition.PCA(n_components=2)
    trans.fit(hidden_pred_train)
    train_dist = trans.transform(hidden_pred_train)
    test_dist = trans.transform(hidden_pred_test)
    
    return train_dist, test_dist

## 1D conv model
params = {
        'n_filter': 16,
        'filter_size': 5,
        'input_dropout': 0.0,
        'hidden_layers': 3,
        'hidden_units': 64,
        'embedding_out_dim': 4,
        'hidden_activation': 'relu', 
        'hidden_dropout': 0.04,
        'gauss_noise': 0.01,
        'norm_type': 'batch', # layer
        'optimizer': {'type': 'adam', 'lr': 1e-4},
        'batch_size': 32,
        'epochs': 80
    }
def create_model(input_len, params):
    
    n_filter = params['n_filter']
    filter_size = params['filter_size']
    
    _input = layers.Input((input_len, 1))

    x = layers.Conv1D(n_filter, filter_size, padding="same")(_input)
    x = layers.Activation(params["hidden_activation"])(x)
    if params['norm_type'] == 'batch':
        x = layers.BatchNormalization()(x)
    elif params['norm_type'] == 'layer':
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(params["hidden_dropout"])(x)
    x = layers.GaussianNoise(params['gauss_noise'])(x)
    
    x = layers.Conv1D(n_filter, filter_size, padding="same")(x)
    x = layers.Activation(params["hidden_activation"])(x)
    if params['norm_type'] == 'batch':
        x = layers.BatchNormalization()(x)
    elif params['norm_type'] == 'layer':
        x = layers.LayerNormalization()(x)
    x = layers.Dropout(params["hidden_dropout"])(x)
    x = layers.GaussianNoise(params['gauss_noise'])(x)
    
    x = layers.MaxPool1D()(x)
    
    for i in range(params['hidden_layers'] - 1):
        x = layers.Conv1D(n_filter * 2, filter_size, padding="same")(x)
        x = layers.Activation(params["hidden_activation"])(x)
        if params['norm_type'] == 'batch':
            x = layers.BatchNormalization()(x)
        elif params['norm_type'] == 'layer':
            x = layers.LayerNormalization()(x)
        x = layers.Dropout(params["hidden_dropout"])(x)
        
    ## あとで挙動を確認したいので、この層に名前をつけます
    x = layers.GlobalMaxPool1D(name="hidden")(x)
    x = layers.Dropout(params["hidden_dropout"])(x)

    ## 0 or 1を当てるのでsigmoidを使います
    _out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(_input, _out, name="Sequential")

    return model

model = create_model(L, params)
model.summary()

### Fitting ###
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=116)
oof = np.zeros(X_train.shape[0])
test_ = np.ones(X_test.shape[0])
cv = 0
cnn_features_tr = np.zeros((X_train.shape[0], 2))
cnn_features_ts = np.zeros((X_test.shape[0], 2))
weights = {0:4, 1:100}

for train_idx, test_idx in skf.split(X_train, y_train[:,0]):

    print("#"*50)
    print("CV={}".format(cv+1))

    X_skf_train = X_train[train_idx, :, :]
    y_skf_train = y_train[train_idx,:]
    X_skf_test = X_train[test_idx,:, :]
    y_skf_test = y_train[test_idx,:]

    ## CVごとにモデル作成
    model = create_model(L, params)

    # compile
    loss = "binary_crossentropy"
    opt = optimizers.Adam(lr=params['optimizer']['lr'])
    model.compile(loss=loss, optimizer=opt)

    # callbacks
    er = callbacks.EarlyStopping(patience=8, min_delta=params['optimizer']['lr'], restore_best_weights=True, monitor='val_loss')
    ReduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, min_delta=params['optimizer']['lr'], mode='min')
    ckp = callbacks.ModelCheckpoint(filepath=output_path + 'best_weight_cv_{}.h5'.format(cv),
                          verbose=1, save_best_only=True)    
    
    # fit
    history = model.fit(X_skf_train, y_skf_train, class_weight=weights, callbacks=[er, ReduceLR, ckp],
                        epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=[X_skf_test, y_skf_test])

    ## 中間ベクトルを取り出すためのモデル
    pred_model = create_model(L, params)
    pred_model.load_weights(output_path + 'best_weight_cv_{}.h5'.format(cv))
    
    ## 中間ベクトル
    train_dist, test_dist = extra_dist(pred_model, X_skf_test, X_test)
    cnn_features_tr[test_idx, :] = train_dist
    cnn_features_ts += test_dist / k

    ## predict
    oof[test_idx] = model.predict(X_skf_test)[:,0]    
    test_ *= model.predict(X_test)[:,0]

    # 後処理
    print("pr-auc: {}".format(metrics.average_precision_score(y_skf_test[:,0], oof[test_idx])))
    cv += 1
    print("#"*50)

### CV ###
prauc = metrics.average_precision_score(y_train[:orig_trlen], oof[:orig_trlen])
ll = metrics.log_loss(y_train[:orig_trlen], oof[:orig_trlen])
lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_train[:orig_trlen], oof[:orig_trlen])
print(f"Overall PR-AUC = {prauc}, LogLoss = {ll}")
print(metrics.classification_report(y_true=y_train[:orig_trlen], y_pred=np.round(oof[:orig_trlen])))

### save CNN features ###
sub["target"] = test_ ** (1 / k)
sub.to_csv(output_path + 'submission_cnn.csv', index=False)
print("submitted!")
np.save(output_path + "cnn_features_tr", cnn_features_tr)
np.save(output_path + "cnn_features_ts", cnn_features_ts)
print("cnn featuers saved!")