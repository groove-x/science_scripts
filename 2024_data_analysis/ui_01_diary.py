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
        self.root.title("ui_01_daily")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x300")

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 00_data_base", command=self.select_input_folder)
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
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder

        for csv_file in diary_csv_files:
            file_path = os.path.join(diary_csv_path, csv_file)
            #df_data = pd.read_csv(file_path)
            df_data = pd.read_csv(file_path, sep=',')
            #df_score2 = df_score[["GhostID", "BirthDate", "q_Date"]]
            df_data = pd.merge(df_data, df_score, how='left', left_on='GhostID', right_on='GhostID')

            #if "EventDate" not in df_data.columns :
            df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
            #df_data['EventDate'] = df_data['Timestamp']
            #df_data["EventDate"] = pd.to_datetime(df_data["Timestamp"], format="%Y-%m-%d")
            df_data['EventDate'] = df_data['Timestamp'].dt.date
            print(df_data)
            #df_data["EventDate"] = pd.to_datetime(df_data["Timestamp"], format="%Y-%m-%d")
            #df_data["EventDate"] = pd.to_datetime(df_data["Timestamp"], format="%Y-%m-%d")
                # https://www.yutaka-note.com/entry/pandas_datetime
                #print(df_data['EventDate'])
            #else:
            #df_data['EventDate'] = pd.to_datetime(df_data['EventDate'])
            #df_data['BirthDate'] = pd.to_datetime(df_data['BirthDate'])
            df_data['BirthDate'] = pd.to_datetime(df_data['BirthDate']).dt.date

            # 日数の差を計算する
            df_data['usedays'] = (pd.to_datetime(df_data['EventDate']) - pd.to_datetime(df_data['BirthDate'])).dt.days
            #df_data['usedays'] = (df_data['EventDate'] - df_data['BirthDate']).apply(lambda x: x.days)
            #df_data['usedays'] = (df_data['EventDate'] - df_data['BirthDate']).dt.days
            print(df_data)
            # usedays
            unique_counts = df_data.groupby(['GhostID', 'Event', 'EventDate', 'usedays', 'q_Date', 'BirthDate']).size().reset_index(name='counts')
            print(unique_counts)
            #unique_counts.plot(kind = 'scatter', x = 'usedays', y = 'counts')
            #plt.show()
            #plt.scatter(x=unique_counts['usedays'], y=unique_counts['counts'])
            #plt.show()
            #unique_counts.to_excel("./data_daily/daily_{}_usedays.xlsx".format(csv_file))
            unique_counts.to_csv("{}/daily_{}_usedays.csv".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

            print(csv_file)
            if "HUGGED_"in csv_file:
                df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
                df_data['Timestamp_end'] = pd.to_datetime(df_data['Timestamp_end'])
                df_data['duration_1'] = df_data['Timestamp_end'] - df_data['Timestamp']
                df_data['duration'] = df_data['duration_1'].map(lambda x: x.total_seconds())
                # https://qiita.com/c60evaporator/items/31fa5e57f1ab64a59b4d
                #df_data['duration'] = (df_data['Timestamp_end'] - df_data['Timestamp']).dt.seconds
                print(df_data)
                unique_duration = df_data.groupby(['GhostID', 'Event', 'EventDate', 'usedays'])["duration"].sum().reset_index().rename(columns={"duration":"sum_duration"})
                #unique_duration = df_data.groupby(['GhostID', 'EventDate']).sum()["duration"].reset_index(name='sum_duration')
                # https://qiita.com/Kotan-86/items/f06e96dbc795edf71b42
                print(unique_duration)
                #unique_duration.to_excel("./data_daily/daily_{}_duration.xlsx".format(csv_file))
                unique_duration.to_csv("{}/daily_{}_duration.csv".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

            if "STROKE_" in csv_file:
                df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
                df_data['Timestamp_end'] = pd.to_datetime(df_data['Timestamp_end'])
                df_data['duration_1'] = df_data['Timestamp_end'] - df_data['Timestamp']
                df_data['duration'] = df_data['duration_1'].map(lambda x: x.total_seconds())
                # https://qiita.com/c60evaporator/items/31fa5e57f1ab64a59b4d
                #df_data['duration'] = (df_data['Timestamp_end'] - df_data['Timestamp']).dt.seconds
                print(df_data)
                unique_duration = df_data.groupby(['GhostID', 'Event', 'EventDate', 'usedays'])["duration"].sum().reset_index().rename(columns={"duration":"sum_duration"})
                #unique_duration = df_data.groupby(['GhostID', 'EventDate']).sum()["duration"].reset_index(name='sum_duration')
                # https://qiita.com/Kotan-86/items/f06e96dbc795edf71b42
                print(unique_duration)
                #unique_duration.to_excel("./data_daily/daily_{}_duration.xlsx".format(csv_file))
                unique_duration.to_csv("{}/daily_{}_duration.csv".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない
        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

