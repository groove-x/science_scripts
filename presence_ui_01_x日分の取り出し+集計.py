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


class AnalysisApp:
    def __init__(self, root):
        self.root = root
        # プログラムのファイル名を取得して、タイトルに設定
        file_name = os.path.basename(__file__)  # ファイルのフルパスからファイル名を取得
        self.root.title(file_name)  # タイトルをファイル名に設定
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x400")

        # ファイル選択ボタン
        self.select_input_file_btn = tk.Button(root, text="Select q_score File", command=self.select_input_file)
        self.select_input_file_btn.pack(pady=10)

        # 日数
        self.filter_label = tk.Label(root, text="Enter days:")
        self.filter_label.pack(pady=5)

        self.filter_entry = tk.Entry(root)
        self.filter_entry.pack(pady=5)

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 00_data_base", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_file_label = tk.Label(root, text="Input File: Not selected")
        self.input_file_label.pack(pady=5)
        
        self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        self.input_folder_label.pack(pady=5)

        self.output_folder_label = tk.Label(root, text="Output Folder: Not selected")
        self.output_folder_label.pack(pady=5)

        # 実行ボタン
        self.run_btn = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_btn.pack(pady=20)

    def select_input_file(self):
        self.input_file = filedialog.askopenfilename(title="Select the file to analyze", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if self.input_file:
            self.input_file_label.config(text=f"Input File: {os.path.basename(self.input_file)}")
        else:
            self.input_file_label.config(text="Input File: Not selected")

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
        if not self.input_folder or not self.output_folder or not self.input_file:
            messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            return
        
        # 入力された文字列を取得
        cal_days = self.filter_entry.get().strip()
        # cal_daysをintに変換する
        cal_days = int(cal_days)  # もしくはfloat(cal_days)

        if not cal_days:
            messagebox.showwarning("Warning", "Please enter a filter text.")
            return


        # ここで実際の解析処理を実行します
        file_score = self.input_file

        df_score = pd.read_csv(file_score, sep=',')
        print(df_score)
        df_score["q_Date"] = pd.to_datetime(df_score["q_Date"]).dt.date

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

            df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
            df_data['EventDate'] = df_data['Timestamp'].dt.date
            print(df_data)

            # DataFrameA の 'A' 列を基準にマージ（結合）する
            df_merged = pd.merge(df_data, df_score, on='GhostID', how='left')
            #df_merged = pd.merge(df_data, df_score, on='GhostID', suffixes=('_B', '_A'))

            # B列の値に基づいてフィルタリング（過去28日間のデータ）
            df_filtered = df_merged[
                (df_merged['EventDate'] > df_merged['q_Date'] - pd.Timedelta(days=cal_days)) & 
                (df_merged['EventDate'] <= df_merged['q_Date'])
            ]

            print(df_filtered)
            # A列に5を掛ける
            df_filtered['Human_min'] = df_filtered['HumanPresence'] * 5
            # https://qiita.com/c60evaporator/items/31fa5e57f1ab64a59b4d
            #df_data['duration'] = (df_data['Timestamp_end'] - df_data['Timestamp']).dt.seconds
            print(df_filtered)
            unique_duration = df_filtered.groupby(['q_Date+GhostID', 'GhostID', 'UseDays', 'q_Date', 'BirthDate', 'score'])["Human_min"].sum().reset_index().rename(columns={"Human_min":"count"})
            #unique_duration = df_data.groupby(['GhostID', 'EventDate']).sum()["duration"].reset_index(name='sum_duration')
            # https://qiita.com/Kotan-86/items/f06e96dbc795edf71b42
            df_b_selected = unique_duration[['q_Date+GhostID', 'count']]  # 'A' 列と 'Data' 列を選択
            # DataFrameA の 'A' 列を基準に左結合
            df_result = pd.merge(df_score, df_b_selected, on='q_Date+GhostID', how='left')
            print(df_result)
            #unique_duration.to_excel("./data_daily/daily_{}_duration.xlsx".format(csv_file))
            # 'count' 列の NaN を 0 に置き換える
            df_result.to_csv("{}/{}days_presence_{}.csv".format(out_folder, cal_days, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない
        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

