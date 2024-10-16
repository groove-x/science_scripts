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
def calculate_change(df, category_value, base_days, a):
    # 基準となる行を取得する
    baseline_value = df.loc[(df['GhostID'] == category_value) & (df['30days_unit'] == base_days), a].values
    if len(baseline_value) == 0:
        return pd.Series([None] * len(df[df['GhostID'] == category_value]), index=df[df['GhostID'] == category_value].index)
    baseline_value = baseline_value[0]
    
    # 変化率を計算する
    percentage_change = df.loc[df['GhostID'] == category_value, a].apply(lambda x: ((x - baseline_value) / baseline_value) * 100)
    return percentage_change



class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ui_13_monthly_cumulative")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x300")

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 03_monthly", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        self.input_folder_label.pack(pady=5)

        self.output_folder_label = tk.Label(root, text="Output Folder: Not selected")
        self.output_folder_label.pack(pady=5)

        # 実行ボタン
        self.run_btn = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=20)

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory(title="Select the folder to analyze")
        if self.input_folder:
            self.input_folder_label.config(text=f"Input Folder: {os.path.basename(self.input_folder)}")
        else:
            self.input_folder_label.config(text="Input Folder: Not selected")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory(title="Select the folder to save results")
        if self.output_folder:
            self.output_folder_label.config(text=f"Output Folder: {os.path.basename(self.output_folder)}")
        else:
            self.output_folder_label.config(text="Output Folder: Not selected")

    def run_analysis(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            return
        
        # ここで実際の解析処理を実行します
        file_score = "GhostID-Birthday-score"

        df_score = pd.read_csv("{}.csv".format(file_score), sep=',')
        print(df_score)

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.xlsx')]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder

        for csv_file in diary_csv_files:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            csv_file=csv_file[:-5]
            csv_file = csv_file[14:]
            #df_data = pd.read_csv(file_path, sep=',')
            df_data = pd.read_excel(file_path)
            #print(df_data)
            plt.figure()

            #df_score2 = df_score[["GhostID", "BirthDate", "score", "q_Date"]]
            #df_data = pd.merge(df_data, df_score2, how='left', left_on='GhostID', right_on='GhostID')
            #print(df_data)

            #df_data["30days_unit"] = (df_data['usedays']//30)*30
            # datetime列をdatetime型に変換する
            #df_data['BirthDate'] = pd.to_datetime(df_data['BirthDate'])

            #if "usedays"in csv_file:
                #unique = df_data.groupby(['GhostID', '30days_unit', 'BirthDate', 'score'])["counts"].sum().reset_index().rename(columns={"counts":"30days_sum"})
            #else:
                #unique = df_data.groupby(['GhostID', '30days_unit', 'BirthDate', 'score'])["sum_duration"].sum().reset_index().rename(columns={"sum_duration":"30days_sum"})

            # 別列の数値を日数として追加する
            #unique['30days_EventDate'] = unique['BirthDate'] + pd.to_timedelta(unique['30days_unit'], unit='d')

            # A列でグループ化し、B列でソート
            unique = df_data.sort_values(by=['GhostID', '30days_unit'])

            # C列の累積和を計算し、D列に保存
            unique['PTD'] = unique.groupby('GhostID')['30days_sum'].cumsum()
            #PTD (period to date, 期間累計)

            # 各GhostIDについて変化量を計算し、新しい列 'change' を作成する
            unique['change_360'] = unique.groupby('GhostID').apply(lambda x: calculate_change(unique, x.name, 360, 'PTD')).reset_index(level=0, drop=True)
            # 各GhostIDについて変化量を計算し、新しい列 'change' を作成する
            unique['change_0'] = unique.groupby('GhostID').apply(lambda x: calculate_change(unique, x.name, 0, 'PTD')).reset_index(level=0, drop=True)

            # 列 'B' を削除
            unique = unique.drop(columns=['30days_sum'])

            # 列名 'A' を 'Alpha' に変更
            unique = unique.rename(columns={'PTD': '30days_sum'})

            csv_file=csv_file[:-4]
            unique.to_excel("{}/monthly_cumulative_{}.xlsx".format(out_folder, csv_file))
            #unique.to_csv("./data_monthly/monthly_{}.csv".format(csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

