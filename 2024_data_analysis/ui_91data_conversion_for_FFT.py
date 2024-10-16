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

# FFT用のデータを作成（データのない部分は0でデータを埋める）
def process_data(diary_csv_path, output_folder, f_name, freq):
    global new_df
    new_df = pd.DataFrame()

    # フォルダ内の全ファイル名を取得
    diary_csv_list = os.listdir(diary_csv_path)
    diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]

    out_folder = os.path.join(output_folder, "01_{}_data_for_FFT".format(f_name))
    os.makedirs(out_folder, exist_ok=True)

    for csv_file in diary_csv_files:
        file_path = os.path.join(diary_csv_path, csv_file)
        #df_data = pd.read_csv(file_path)
        df_data = pd.read_csv(file_path, sep=',')
        df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
        #df_score2 = df_score[["GhostID", "BirthDate", "q_Date"]]
        #df_data = pd.merge(df_data, df_score, how='left', left_on='GhostID', right_on='GhostID')
        event_name = df_data['Event'].unique()
        event_str = event_name[0]
        df = pd.DataFrame(df_data)

        # Timestamp列を時間単位に変換
        if freq == 'M':
            #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 30).apply(lambda x: pd.offsets.DateOffset(days=x))
            # 時間部分を00:00:00にリセット
            df['Timestamp'] = df['Timestamp'].dt.normalize()
            df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
            #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
        elif freq == 'W':
            df['Timestamp'] = df['Timestamp'].dt.normalize()
            df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 7, unit='D')
            #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
            #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
        else:
            df['Timestamp'] = df['Timestamp'].dt.floor(freq)

        #df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor(freq)

        # IDとTimestampでグループ化して行数をカウント
        grouped = df.groupby(['GhostID', 'Timestamp']).size().reset_index(name='sum')

        # 各グループごとに飛び値のある時間を埋める
        new_df = pd.DataFrame()

        for name, group in grouped.groupby('GhostID'):
            # 時間範囲の作成
            if freq == 'M':
                full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='30D')
            elif freq == 'W':
                full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='7D')
                #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
            else:
                full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq=freq)
            
            # full_time_rangeに基づいて再インデックス化し、欠損時間を埋める
            group = group.set_index('Timestamp').reindex(full_time_range, fill_value=0).reset_index()
            group['GhostID'] = name
            group.columns = ['Timestamp', 'GhostID','sum']
            
            # 新しいデータフレームに追加
            new_df = pd.concat([new_df, group], ignore_index=True)
        
        new_df['Event']=event_str
        new_df.to_csv("{}/{}_{}_usedays.csv".format(out_folder, f_name, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

        print(csv_file)
        if "HUGGED_"in csv_file:
            df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
            df_data['Timestamp_end'] = pd.to_datetime(df_data['Timestamp_end'])
            df_data['duration_1'] = df_data['Timestamp_end'] - df_data['Timestamp']
            df_data['sum'] = df_data['duration_1'].map(lambda x: x.total_seconds())
            print(df_data)

            df = df_data[['Timestamp', 'sum', 'GhostID']]

            # Timestamp列を時間単位に変換
            if freq == 'M':
                #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 30).apply(lambda x: pd.offsets.DateOffset(days=x))
                # 時間部分を00:00:00にリセット
                df['Timestamp'] = df['Timestamp'].dt.normalize()
                df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
                #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
            elif freq == 'W':
                df['Timestamp'] = df['Timestamp'].dt.normalize()
                df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 7, unit='D')
                #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
                #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
            else:
                df['Timestamp'] = df['Timestamp'].dt.floor(freq)
            #df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor(freq)

            # 'ID'ごとにデータをグループ化して処理
            grouped = df.groupby(['GhostID', 'Timestamp']).sum().reset_index()

            # 各グループごとに飛び値のある時間を埋める
            new_df = pd.DataFrame()

            for name, group in grouped.groupby('GhostID'):
                # 時間範囲の作成
                if freq == 'M':
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='30D')
                elif freq == 'W':
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='7D')
                    #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
                else:
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq=freq)
                
                # full_time_rangeに基づいて再インデックス化し、欠損時間を埋める
                group = group.set_index('Timestamp').reindex(full_time_range, fill_value=0).reset_index()
                group['GhostID'] = name
                group.columns = ['Timestamp', 'GhostID','sum']
                
                # 新しいデータフレームに追加
                new_df = pd.concat([new_df, group], ignore_index=True)

            new_df['Event'] = "HUGGED_duration"
            new_df.to_csv("{}/{}_{}_duration.csv".format(out_folder, f_name, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

        elif "STROKE_" in csv_file:
            df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])
            df_data['Timestamp_end'] = pd.to_datetime(df_data['Timestamp_end'])
            df_data['duration_1'] = df_data['Timestamp_end'] - df_data['Timestamp']
            df_data['sum'] = df_data['duration_1'].map(lambda x: x.total_seconds())

            df = df_data[['Timestamp', 'sum', 'GhostID']]

            # Timestamp列を時間単位に変換
            if freq == 'M':
                #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 30).apply(lambda x: pd.offsets.DateOffset(days=x))
                # 時間部分を00:00:00にリセット
                df['Timestamp'] = df['Timestamp'].dt.normalize()
                df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
                #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
            elif freq == 'W':
                df['Timestamp'] = df['Timestamp'].dt.normalize()
                df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 7, unit='D')
                #df['Timestamp'] = df['Timestamp'] - pd.to_timedelta((df['Timestamp'].dt.day - 1) % 30, unit='D')
                #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
            else:
                df['Timestamp'] = df['Timestamp'].dt.floor(freq)
            #df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.floor(freq)

            # 'ID'ごとにデータをグループ化して処理
            grouped = df.groupby(['GhostID', 'Timestamp']).sum().reset_index()

            # 各グループごとに飛び値のある時間を埋める
            new_df = pd.DataFrame()

            for name, group in grouped.groupby('GhostID'):
                # 時間範囲の作成
                if freq == 'M':
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='30D')
                elif freq == 'W':
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='7D')
                    #df.loc[:, 'Timestamp'] = df['Timestamp'] - (df['Timestamp'].dt.day % 7).apply(lambda x: pd.offsets.DateOffset(days=x))
                else:
                    full_time_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq=freq)
                
                # full_time_rangeに基づいて再インデックス化し、欠損時間を埋める
                group = group.set_index('Timestamp').reindex(full_time_range, fill_value=0).reset_index()
                group['GhostID'] = name
                group.columns = ['Timestamp', 'GhostID','sum']
                
                # 新しいデータフレームに追加
                new_df = pd.concat([new_df, group], ignore_index=True)

            new_df['Event'] = "STROKE_duration"
            new_df.to_csv("{}/{}_{}_duration.csv".format(out_folder, f_name, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ui_91_data_conversion_for_FFT")
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
        df_score['BirthDate'] = pd.to_datetime(df_score['BirthDate']).dt.date

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        output_folder = self.output_folder

        process_data(diary_csv_path, output_folder, 'hourly', 'H')
        process_data(diary_csv_path, output_folder, 'daily', 'D')
        process_data(diary_csv_path, output_folder, 'weekly', 'W')
        process_data(diary_csv_path, output_folder, 'monthly', 'M')

        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

