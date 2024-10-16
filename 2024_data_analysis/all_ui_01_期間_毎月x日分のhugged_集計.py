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
import matplotlib.pyplot as plt

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
        #self.select_input_file_btn = tk.Button(root, text="Select event File", command=self.select_input_file)
        #self.select_input_file_btn.pack(pady=10)

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: event", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)


        # 日数
        self.filter_label = tk.Label(root, text="Enter days:")
        self.filter_label.pack(pady=5)

        self.filter_entry = tk.Entry(root)
        self.filter_entry.pack(pady=5)


        # フォルダパス表示ラベル
        #self.input_file_label = tk.Label(root, text="Input File: Not selected")
        #self.input_file_label.pack(pady=5)
        
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
        if not self.input_folder or not self.output_folder:
        #if not self.input_folder or not self.output_folder or not self.input_file:
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
        #file_score = self.input_file

        #df_score = pd.read_csv(file_score, sep=',')
        #print(df_score)
        #df_score["q_Date"] = pd.to_datetime(df_score["q_Date"]).dt.date
        #df_score["hugged_hr"] = df_score["duration"]/3600

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]
        #print(diary_csv_files)

        df_merged = pd.concat([pd.read_csv(os.path.join(diary_csv_path, file), sep=',') for file in diary_csv_files], ignore_index=True)

        out_folder = self.output_folder

        #for csv_file in diary_csv_files:
            #file_path = os.path.join(diary_csv_path, csv_file)
            #df_data = pd.read_csv(file_path)
            #df_data = pd.read_csv(file_path, sep=',')

        df_merged['Timestamp'] = pd.to_datetime(df_merged['Timestamp'])
        df_merged['EventDate'] = df_merged['Timestamp'].dt.date
        print(df_merged)

        # Timestamp列をdatetimeに変換
        #df_merged['Timestamp'] = pd.to_datetime(df_merged['Timestamp'])

        # 1日から28日までの日付でフィルタリング
        #df_filtered = df_merged[df_merged['Timestamp'].dt.day <= cal_days]
        df_filtered = df_merged[df_merged['Timestamp'].dt.day <= cal_days]

        # 月ごと、かつIDごとの発生回数と発生日数を集計
        monthly_stats = df_filtered.groupby([pd.Grouper(key='Timestamp', freq='M'), 'GhostID']).agg(
            occurrences=('GhostID', 'size'),  # 発生回数
            unique_days=('Timestamp', lambda x: x.dt.date.nunique())  # 発生日数
        ).reset_index()

        print(monthly_stats)
        monthly_stats.to_excel('{}/monthly_stats.xlsx'.format(out_folder))

        # 発生日数が25日以上の行をフィルタリング
        filtered_stats = monthly_stats[monthly_stats['unique_days'] >= 25]
        filtered_stats.to_excel('{}/filtered_stats.xlsx'.format(out_folder))

        # 発生回数の統計情報を計算
        monthly_summary = filtered_stats.groupby(pd.Grouper(key='Timestamp', freq='M')).agg(
        #monthly_summary = filtered_stats.groupby(pd.Grouper(level='Timestamp')).agg(
            id_count=('GhostID', 'nunique'),  # ID数
            max_occurrences=('occurrences', 'max'),
            min_occurrences=('occurrences', 'min'),
            median_occurrences=('occurrences', 'median'),
            q75_occurrences=('occurrences', lambda x: x.quantile(0.75)),
            q80_occurrences=('occurrences', lambda x: x.quantile(0.80)),
            mean_occurrences=('occurrences', 'mean')
        ).reset_index()

        # 結果をCSVファイルに出力
        monthly_summary.to_excel('{}/monthly_summary.xlsx'.format(out_folder))

        # 集計したい特定の期間を定義 (例: 2023年4月1日から2024年7月31日まで)
        start_date = pd.to_datetime('2023-04-01')  # 日付をdatetime型に変換
        end_date = pd.to_datetime('2024-07-31')

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        

        # 特定の期間でフィルタリング
        df_filtered = df_merged[df_merged['Timestamp'].dt.day <= cal_days]
        df_filtered_time = df_filtered[(df_filtered['Timestamp'] >= start_date) & (df_filtered['Timestamp'] <= end_date)]
        print(df_filtered)

        # 月ごと、かつIDごとの発生回数と発生日数を集計
        monthly_stats = df_filtered_time.groupby([pd.Grouper(key='Timestamp', freq='M'), 'GhostID']).agg(
            occurrences=('GhostID', 'size'),  # 発生回数
            unique_days=('Timestamp', lambda x: x.dt.date.nunique())  # 発生日数
        ).reset_index()

        print(monthly_stats)
        monthly_stats.to_csv('{}/monthly_stats_{}-{}.csv'.format(out_folder, start_date_str, end_date_str), index=False)

        # 発生日数が25日以上の行をフィルタリング
        filtered_stats = monthly_stats[monthly_stats['unique_days'] >= 25]
        #filtered_stats.to_excel('{}/filtered_stats.xlsx'.format(out_folder))

        # インデックスをリセット
        filtered_stats = filtered_stats.reset_index(drop=True)

        # 重複インデックスの確認と処理
        if not filtered_stats.index.is_unique:
            print("Warning: 重複インデックスが検出されました。重複を削除します。")
            print("重複インデックスの数:", filtered_stats.index.duplicated().sum())
            print("重複インデックスの内容:\n", filtered_stats[filtered_stats.index.duplicated(keep=False)])

            # 重複インデックスを削除
            filtered_stats = filtered_stats.loc[~filtered_stats.index.duplicated(keep='first')]

        print(filtered_stats.index)
        if not filtered_stats.index.is_unique:
            print("Warning: 重複インデックスが検出されました。")
        else:
            print("重複インデックスはありません。")

        print(monthly_stats[monthly_stats.duplicated(subset=['Timestamp', 'GhostID'], keep=False)])

        # 統計情報を計算（特定期間に対して）
        summary_stats = filtered_stats.agg(
            #id_count=('GhostID', 'nunique'),  # ID数
            row_count=('occurrences', 'size'),  # 行数（レコード数）
            max_occurrences=('occurrences', 'max'),
            min_occurrences=('occurrences', 'min'),
            median_occurrences=('occurrences', 'median'),
            q75_occurrences=('occurrences', lambda x: x.quantile(0.75)),
            q80_occurrences=('occurrences', lambda x: x.quantile(0.80)),
            mean_occurrences=('occurrences', 'mean')
        )

        # 結果をCSVファイルに出力
        summary_stats.to_excel('{}/summary_{}_{}.xlsx'.format(out_folder, start_date_str, end_date_str))

        # ヒストグラムの描画（発生回数 'occurrences' 列に対して）
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_stats['occurrences'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Occurrences Histogram')
        plt.xlabel('Occurrences')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('{}/occurrences_histogram.png'.format(out_folder))  # ヒストグラムを保存
        plt.show()

        # 度数分布表（'occurrences' 列に対する度数分布を求める）
        frequency_distribution = filtered_stats['occurrences'].value_counts().sort_index()
        print(frequency_distribution)

        # 度数分布表をExcelに保存
        frequency_distribution.to_excel('{}/occurrences_frequency_{}-{}.xlsx'.format(out_folder, start_date_str, end_date_str))

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

