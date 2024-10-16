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


class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analysis Tool")
        self.input_folder = ""
        self.output_folder = ""

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # 実行ボタン
        self.run_btn = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=20)

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory(title="Select the folder to analyze")
        if self.input_folder:
            print(f"Selected input folder: {self.input_folder}")
        else:
            print("No input folder selected")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory(title="Select the folder to save results")
        if self.output_folder:
            print(f"Selected output folder: {self.output_folder}")
        else:
            print("No output folder selected")

    def run_analysis(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            return
        
        # ここで実際の解析処理を実行します
        #file_score = "GhostID-Birthday-score"

        #df_score = pd.read_csv("{}.csv".format(file_score), sep=',')
        #print(df_score)

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder
        df_out = pd.DataFrame()

        for csv_file in diary_csv_files:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            df_data = pd.read_csv(file_path)
            print(df_data)
            df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])

            # 列 'Timestamp' の最小値を持つ行を取得
            min_row = df_data[df_data['Timestamp'] == df_data['Timestamp'].min()]
            df_out = pd.concat([df_out, min_row], axis=0, ignore_index=True)
            print(df_out)

        csv_file=csv_file[:-4]

        df_out.to_excel("{}/min_timestamp.xlsx".format(out_folder))

        messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

