"""
performs feature engineering

:USAGE:
>>> python feature_engineering.py

"""

### Libraries ###
import numpy as np
import pandas as pd
import os, sys
import umap
from sklearn import decomposition
from scipy.signal import find_peaks, peak_widths

### data location ###
input_path = "../input/"
output_path = "../output/"

### load data ###
def read_data():
    # load
    train = pd.read_csv(input_path + "train.csv")
    test = pd.read_csv(input_path + "test.csv")
    fitting = pd.read_csv(input_path + "fitting__fixed.csv")   
    submission = pd.read_csv(input_path + "atmaCup5__sample_submission.csv")
    
    # merge
    train = train.merge(fitting, how='left', on='spectrum_id')
    test = test.merge(fitting, how='left', on='spectrum_id')
    return train, test, submission
train, test, sub = read_data()

### feature engineering ###
## params features
def fitparams_features(df):
    df['params1_4_ratio'] = df['params1'] / df['params4']
    df['params1_3_ratio'] = df['params1'] / df['params3']
    df['params4_6_ratio'] = df['params4'] / df['params6']
    df['params13_46_ratio'] = df['params1_3_ratio'] / df['params4_6_ratio']
    return df
train = fitparams_features(train)
test = fitparams_features(test)

## params relative to mean of positive labels
def parampos_features(train, test):    
    # diff to mean
    for f in [f"params{i}" for i in [2, 3, 5, 6]]:
        # to mean
        me = train.loc[train['target'] == 1, f].mean()
        train[f'{f}_to_posmean'] = np.abs(train[f].values - me)
        test[f'{f}_to_posmean'] = np.abs(test[f].values - me)

        # to median
        me = train.loc[train['target'] == 1, f].median()
        train[f'{f}_to_posmedian'] = np.abs(train[f].values - me)
        test[f'{f}_to_posmedian'] = np.abs(test[f].values - me)
    return train, test
train, test = parampos_features(train, test)

## wave features
def wave_features(spec_df):
    d = {}
    
    # peak features
    peaks, properties = find_peaks(spec_df['y'], prominence=800, width=2)
    d['num_peaks'] = len(peaks)
    if d['num_peaks'] == 0:
        d['prominences'] = 0
        d['widths'] = 0
        d['width_heights'] = 0
        d['peak_pos'] = 0
    else:
        peak_idx = np.argmax(properties['prominences'])
        d['prominences'] = properties['prominences'][peak_idx]
        d['widths'] = properties['widths'][peak_idx]
        d['width_heights'] = properties['width_heights'][peak_idx]
        d['peak_pos'] = peaks[peak_idx]    
    
    # basic stats
    sig = spec_df['y'].values
    grad = np.gradient(spec_df['y'].values)
    grad2 = np.gradient(grad)
    grad3 = np.gradient(grad2)
    names = ['sig', 'grad', 'grad2', 'grad3',]
    for n, s in zip(names, [sig, grad, grad2, grad3, ]):
        d[f'{n}_mean'] = np.mean(s)
        d[f'{n}_std'] = np.std(s)
        d[f'{n}_max'] = np.max(s)
        d[f'{n}_min'] = np.min(s)
        d[f'{n}_absmin'] = np.abs(d[f'{n}_min'])
        d[f'{n}_max2min'] = d[f'{n}_max'] - d[f'{n}_min']
        d[f'{n}_argmax'] = np.argmax(s)
        d[f'{n}_argmin'] = np.argmin(s)
        d[f'{n}_argmax2min'] = np.abs(d[f'{n}_argmax'] - d[f'{n}_argmin'])
        try:
            d[f'{n}_peaknearsum'] = np.sum(s[d[f'{n}_argmax'] - 10 : d[f'{n}_argmax'] + 10]) / d[f'{n}_max']
        except:
            d[f'{n}_peaknearsum'] = 0
        sp = pd.DataFrame(data=s, columns=['a'])
        d[f'{n}_skew'] = sp['a'].skew()
        d[f'{n}_kurt'] = sp['a'].kurtosis()
    
    # to pandas
    d = pd.DataFrame(d, index=[0])
    return d

def add_wavefeats(df):
    # waveform features
    l = 511
    wave_df = pd.DataFrame()
    for i in range(df.shape[0]):
        fname = df['spectrum_filename'].values[i]
        spec_df = pd.read_csv(input_path + f'spectrum_raw/{fname}', sep='\t', header=None)
        spec_df.columns = ['x', 'y']
        d = wave_features(spec_df)
        wave_df = pd.concat([wave_df, d], ignore_index=True)
        if i == 0:
            waves = spec_df['y'].values[:l]
        else:
            waves = np.vstack((waves, spec_df['y'].values[:l]))
    df = pd.concat([df, wave_df], axis=1)    
    return df, waves
train, tr_wave = add_wavefeats(train)
test, ts_wave = add_wavefeats(test)

## FFT features
def add_fft(df, waves):
    df["fft_50_100"] = 0
    for i in range(df.shape[0]):
        # compute psd
        sig = waves[i, :]
        fft = np.fft.rfft(sig)
        psd = np.abs(fft) ** 2
        psd = 20 * np.log10(psd)
        
        # extract features
        df.loc[i, "fft_50_100"] = np.mean(psd[50:100]) - np.mean(psd[100:200])
        
        # psds
        if i == 0:
            psds = psd
        else:
            psds = np.vstack((psds, psd))
    return df, psds
train, tr_psds = add_fft(train, tr_wave)
test, ts_psds = add_fft(test, ts_wave)

## UMAP, PCA features
# normalize
def wave_normalizer(tr_wave, ts_wave):
    me = np.mean(tr_wave[:])
    sd = np.std(tr_wave[:])
    tr_wave = (tr_wave - me) / sd
    ts_wave = (ts_wave - me) / sd
    return tr_wave, ts_wave
tr_wave, ts_wave = wave_normalizer(tr_wave, ts_wave)
tr_psds, ts_psds = wave_normalizer(tr_psds, ts_psds)

# dimensionality reduction
def add_emb(train_wave, test_wave, name="wave", method="umap"):
    # original signal
    if method == "umap":
        em = umap.UMAP(random_state=116)  # (added after the competition)
    elif method == "pca":
        em = decomposition.PCA(n_components=2)
    em.fit(train_wave)
    train_emb = em.transform(train_wave)
    test_emb = em.transform(test_wave)
    for i in range(2):
        train[f'{name}_{method}{i+1}'] = train_emb[:, i]
        test[f'{name}_{method}{i+1}'] = test_emb[:, i]
            
    return train, test
train, test = add_emb(tr_psds, ts_psds, name="fft", method="umap")
train, test = add_emb(tr_psds, ts_psds, name="fft", method="pca")

### save data ###
train.to_csv(output_path + "train.csv", index=False)
test.to_csv(output_path + "test.csv", index=False)
np.save(output_path + "tr_wave", tr_wave)
np.save(output_path + "ts_wave", ts_wave)
print("saved!")
