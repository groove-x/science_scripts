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
from scipy.stats import wilcoxon
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import curve_fit
import tensorflow as tf

# 相関係数とp値を計算する関数
def spearman_corr_p(group, index_name):
    corr, p_value = spearmanr(group[index_name], group['score'])
    return pd.Series({'Spearman_Correlation': corr, 'p_value': p_value})

# 指数関数的減衰モデル
def exp_decreasing(t, a, b):
    return a * np.exp(-b * t)


class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ui_09_fft")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x600")

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 93_for_fft", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        self.input_folder_label.pack(pady=5)

        self.output_folder_label = tk.Label(root, text="Output Folder: Not selected")
        self.output_folder_label.pack(pady=5)

        # 開始日入力
        self.start_date_label = tk.Label(root, text="Enter Start Date (YYYY-MM-DD):")
        self.start_date_label.pack(pady=5)

        self.start_date_entry = tk.Entry(root)
        self.start_date_entry.insert(0, "2021-02-01")  # 初期値を設定
        self.start_date_entry.pack(pady=5)
        
        # item入力
        self.item_label = tk.Label(root, text="何データ分:")
        self.item_label.pack(pady=5)

        self.item_entry = tk.Entry(root)
        self.item_entry.insert(0, "all")  # 初期値を設定
        self.item_entry.pack(pady=5)

        # データ長入力
        self.length_label = tk.Label(root, text="何データ分:")
        self.length_label.pack(pady=5)

        self.length_entry = tk.Entry(root)
        self.length_entry.insert(0, "365")  # 初期値を設定
        self.length_entry.pack(pady=5)

        # データ単位入力
        self.unit_label = tk.Label(root, text="データ単位:")
        self.unit_label.pack(pady=5)

        self.unit_entry = tk.Entry(root)
        self.unit_entry.insert(0, "30")  # 初期値を設定
        self.unit_entry.pack(pady=5)

        # 小数点以下入力
        self.round_label = tk.Label(root, text="小数点以下:")
        self.round_label.pack(pady=5)

        self.round_entry = tk.Entry(root)
        self.round_entry.insert(0, "4")  # 初期値を設定
        self.round_entry.pack(pady=5)


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
            #messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            print("Warning", "Please select both input and output folders before running the analysis.")
            return

        start_date = self.start_date_entry.get()
        if not start_date:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return

        items = self.item_entry.get()
        if not items:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return

        sample_length_str = self.length_entry.get()
        if not sample_length_str:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return
        sample_length = int(sample_length_str)

        sample_unit_str = self.unit_entry.get()
        if not sample_unit_str:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return
        sample_unit = int(sample_unit_str)

        sample_round_str = self.round_entry.get()
        if not sample_round_str:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return
        sample_round = int(sample_round_str)

        # ここで実際の解析処理を実行します
        file_score = "GhostID-Birthday-score"

        df_score = pd.read_csv("{}.csv".format(file_score), sep=',')
        print(df_score)
        df_score['BirthDate'] = pd.to_datetime(df_score['BirthDate'])
        df_score2 = df_score[df_score['BirthDate'] > start_date]

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        # CSVファイルとExcelファイルのファイル名をそれぞれ取得
        diary_files = [file for file in diary_csv_list if file.endswith('.csv') or file.endswith('.xlsx')]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder


        # サンプリングレートを1日に1回（1 Hz）として設定
        #sampling_rate = 1.0  # Hz

        # 小数点以下の桁数
        #decimal_places = 8

        if "all" in items:
            cal_files = diary_files
        else:
            # 2. カンマで区切られた文字列を抽出
            item_list = items.split(',')  # カンマで分割
             # 指定された項目が含まれるファイル名を抽出
            cal_files = [file for file in diary_files if any(item in file for item in item_list)]

        for csv_file in cal_files:
        #for csv_file in diary_csv_files:
            #if "STROKE" in csv_file:
            print(csv_file)
            df_data2 = pd.DataFrame()
            file_path = os.path.join(diary_csv_path, csv_file)
            if file_path.endswith('.csv'):
                # CSVファイルとして読み込む
                df_data2 = pd.read_csv(file_path)
                csv_file=csv_file[:-4]
            elif file_path.endswith('.xlsx'):
                # Excelファイルとして読み込む
                df_data2 = pd.read_excel(file_path)
                csv_file=csv_file[:-5]
            #file_suffix = f"{start_date.replace('-', '')}"  # 日付の区切りを取り除く
            #csv_file=f"{file_suffix}_{csv_file}"
            #df_data = pd.read_csv(file_path, sep=',')
            #df_data2 = pd.read_csv(file_path, sep=',')
            #df_score2 = df_score[["GhostID", "BirthDate"]]
            #df_data2 = pd.merge(df_data, df_score2, how='left', left_on='GhostID', right_on='GhostID')
            print(df_data2)

            # 全ての列名を取得
            column_names = df_data2.columns

            eventdate_str = ""
            unit_str = ""
            sum_str = ""

            if 'Timestamp' in column_names:
                # 列名を変更する
                #df_data2.rename(columns={'GhostID': 'sum', 'sum': 'GhostID'}, inplace=True)
                # ↑hourlyで間違っていたので。（修正済）
                eventdate_str = "Timestamp"
                unit_str = ""
                sum_str = "sum"
            else:
                # 特定の文字列（例えば "Temp"）を含む列名を抽出
                search_str = "EventDate"
                col_str = [col for col in column_names if search_str in col]
                eventdate_str = col_str[0]
                search_str = "unit"
                col_str = [col for col in column_names if search_str in col]
                unit_str = col_str[0]
                search_str = "sum"
                col_str = [col for col in column_names if search_str in col]
                sum_str = col_str[0]

            # 結果を保存するための空のDataFrameを作成
            #fft_results = pd.DataFrame()
            result_df = pd.DataFrame()
            
            # 指定した誕生日以降のデータをdf_data2から抜き出す
            df_data3 = df_data2.merge(df_score2[['GhostID']], on='GhostID')
            df=df_data3[['GhostID', sum_str]]

            # id列でグループ化し、各グループに対してFFTを適用
            for id_value, group in df.groupby('GhostID'):
                signal = group[sum_str].values

                # データが200より多い場合は最初の200データのみ使用
                if len(signal) > sample_length:
                    signal = signal[:sample_length]

                 # FFTを適用
                fft_values = tf.signal.rfft(signal)
                
                # 複素数のまま保存するか、絶対値に変換するかは用途による
                fft_abs = np.abs(fft_values.numpy())

                frequencies = np.fft.rfftfreq(len(signal), d=1)
                # 1単位を？とする
                freq_in = frequencies / sample_unit

                # 周波数を指定の小数点以下の桁数で丸め
                rounded_frequencies = np.round(freq_in, decimals=sample_round)
                
                # 周波数のユニークな値を取得
                unique_frequencies = np.unique(rounded_frequencies)

                # 周波数とFFT結果をDataFrameに変換
                #fft_df = pd.DataFrame(index=freq_in)
                fft_df = pd.DataFrame(index=unique_frequencies)
                
                # 各周波数でのFFT値を設定
                #for i, freq in enumerate(freq_in):
                for i, freq in enumerate(rounded_frequencies):
                    fft_df.loc[freq, id_value] = fft_abs[i]
                #fft_df = pd.DataFrame(fft_abs, index=freq_in_years)
                #fft_df = pd.DataFrame(fft_abs, index=np.fft.rfftfreq(len(signal)))
                #fft_df.columns = [id_value]  # idを列名とする

                # 結果を結合
                if result_df.empty:
                    result_df = fft_df
                else:
                    result_df = pd.concat([result_df, fft_df], axis=1)
                print(fft_df)
                #print(result_df)
                # 結果をマスターDataFrameに追加
                #fft_results = pd.concat([fft_results, fft_df], ignore_index=True)
            
            # index名をfrequencyに設定
            result_df.index.name = 'frequency'

            print(result_df)

            result_df.to_excel(f"{out_folder}/cal_fft_{start_date}_{sample_length}_{sample_unit}_{sample_round}_{csv_file}.xlsx")
            #fft_results.to_csv(f"{out_folder}/cal_fft_{csv_file}.csv", index=False)
            #fft_results.to_excel(f"{out_folder}/fft_{csv_file}.xlsx", index=False)

        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

