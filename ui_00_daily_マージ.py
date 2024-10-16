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
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


# 相関係数とp値を計算する関数
def spearman_corr_p(group, index_name):
    corr, p_value = spearmanr(group[index_name], group['score'])
    return pd.Series({'Spearman_Correlation': corr, 'p_value': p_value})

# 30days_sum列の数値の変化量を計算する関数
def calculate_change(df, category_value, base_days):
    # 基準となる行を取得する
    baseline_value = df.loc[(df['GhostID'] == category_value) & (df['30days_unit'] == base_days), '30days_sum'].values
    if len(baseline_value) == 0:
        return pd.Series([None] * len(df[df['GhostID'] == category_value]), index=df[df['GhostID'] == category_value].index)
    baseline_value = baseline_value[0]
    
    # 変化率を計算する
    percentage_change = df.loc[df['GhostID'] == category_value, '30days_sum'].apply(lambda x: ((x - baseline_value) / baseline_value) * 100)
    return percentage_change

class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ui_00_マージ")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x300")

        # ファイル選択ボタン
        self.select_file1_btn = tk.Button(root, text="Select Input File 1", command=self.select_input_file1)
        self.select_file1_btn.pack(pady=10)

        self.select_file2_btn = tk.Button(root, text="Select Input File 2", command=self.select_input_file2)
        self.select_file2_btn.pack(pady=10)

        # ファイルパス表示ラベル
        self.input_file1_label = tk.Label(root, text="Input File 1: Not selected")
        self.input_file1_label.pack(pady=5)

        self.input_file2_label = tk.Label(root, text="Input File 2: Not selected")
        self.input_file2_label.pack(pady=5)
        # 実行ボタン
        self.run_btn = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=20)

    def select_input_file1(self):
        self.input_file1 = filedialog.askopenfilename(title="Select the first file to analyze", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if self.input_file1:
            self.input_file1_label.config(text=f"Input File 1: {os.path.basename(self.input_file1)}")
        else:
            self.input_file1_label.config(text="Input File 1: Not selected")

    def select_input_file2(self):
        self.input_file2 = filedialog.askopenfilename(title="Select the second file to analyze", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if self.input_file2:
            self.input_file2_label.config(text=f"Input File 2: {os.path.basename(self.input_file2)}")
        else:
            self.input_file2_label.config(text="Input File 2: Not selected")

    def run_analysis(self):
        if not self.input_file1 or not self.input_file2:
            messagebox.showwarning("Warning", "Please select both input files before running the analysis.")
            return
        
                # ここで実際の解析処理を実行します
        df_data1 = pd.read_csv(self.input_file1, sep=',')
        df_data2 = pd.read_csv(self.input_file2, sep=',')
        event1 = df_data1['Event'].unique()
        event2 = df_data2['Event'].unique()

        df_data = pd.concat([df_data1, df_data2], axis=0, ignore_index=True)
        event_str = f'{event1[0]}_and_{event2[0]}'
        df_data['Event'] = event_str


        # ファイル名をつなげて出力ファイル名を生成
        file_name1 = os.path.basename(self.input_file1).replace('.csv', '')
        file_name2 = os.path.basename(self.input_file2).replace('.csv', '')
        output_file_name = f"{event_str}_result.csv"

        # 出力ファイルのパス
        output_folder = os.path.dirname(self.input_file2)
        output_file = os.path.join(output_folder, output_file_name)

        # 結果を保存
        df_data.to_csv(output_file, index=False)

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

