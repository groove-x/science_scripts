import pandas as pd
import os
import sys
import datetime
import numpy as np
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import kstest
from sklearn.metrics import roc_curve, auc, confusion_matrix  # 必要なライブラリをインポート
from scipy.stats import spearmanr


f_q = "05_結合用：2024_uid-ghost_234_bq-results-20240814-090454-1723626313635.csv"
f_g = "05_結合用：20230406-20240712のOH_LOVOTとの生活に関するアンケート（回答）.csv"

df_q = pd.read_csv(f_q, sep=",")
df_g = pd.read_csv(f_g, sep=",")

#df_data = pd.merge(df_d, df_q, on='UID', how='left')
df_data = pd.merge(df_q, df_g, on='UID', how='outer')
df_data.to_excel("merge_outer_q_g.xlsx")



