"""Kaggle Titanic の練習で，複数のノートブックから使うコード．

データの取得・読み込みと提出ファイルの書き出しは titanic.data に，EDA（探索的データ分析）用の
グラフは titanic.visualize にまとめている．pip install -e . でインストールして使う．

  Typical usage example:

  from titanic.data import load_data
  train, test = load_data()
"""
