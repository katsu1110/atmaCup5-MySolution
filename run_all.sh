#!/bin/sh
cd code

# まず特徴量を作ります
python feature_engineering.py

# CNNの特徴量を追加します
python add_cnn_features.py

# GBDTを走らせます（pseudolabelなし）
for gbdt in 'lgb' 'xgb' 'catb'
do
    python run_gbdt.py gbdt 0
done

# 中央値でアンサンブルします
python ensemble.py

# GBDTを走らせます（↑を元にpseudolabel）
for gbdt in 'lgb' 'xgb' 'catb'
do
    python run_gbdt.py gbdt 1
done

# 中央値でアンサンブルします
python ensemble.py

