#!/bin/sh
cd code

# まず特徴量を作ります
python feature_engineering.py

# CNNの特徴量を追加します
python add_cnn_features.py

# GBDTを走らせます（pseudolabelなし）
python run_gbdt.py 'lgb' 0
python run_gbdt.py 'xgb' 0
python run_gbdt.py 'catb' 0

# 中央値でアンサンブルします
python ensemble.py

# GBDTを走らせます（↑を元にpseudolabel）
python run_gbdt.py 'lgb' 1
python run_gbdt.py 'xgb' 1
python run_gbdt.py 'catb' 1

# 中央値でアンサンブルします
python ensemble.py

