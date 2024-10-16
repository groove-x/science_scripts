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
        self.root.title("ui_93_xdays_FFT向け")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x400")

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 03_", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        self.input_folder_label.pack(pady=5)

        self.output_folder_label = tk.Label(root, text="Output Folder: Not selected")
        self.output_folder_label.pack(pady=5)

        # 開始日入力
        #self.start_date_label = tk.Label(root, text="Enter Start Date (YYYY-MM-DD):")
        #self.start_date_label.pack(pady=5)

        #self.start_date_entry = tk.Entry(root)
        #self.start_date_entry.insert(0, "2021-02-01")  # 初期値を設定
        #self.start_date_entry.pack(pady=5)
        
        # item入力
        #self.item_label = tk.Label(root, text="Enter Items:")
        #self.item_label.pack(pady=5)

        #self.item_entry = tk.Entry(root)
        #self.item_entry.insert(0, "all")  # 初期値を設定
        #self.item_entry.pack(pady=5)

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

        # ここで実際の解析処理を実行します
        file_score = "GhostID-Birthday-score"

        df_score = pd.read_csv("{}.csv".format(file_score), sep=',')
        print(df_score)

        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.xlsx')]

        out_folder = self.output_folder

        for csv_file in diary_csv_files:
            #if "STROKE" in csv_file:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            csv_file=csv_file[:-5]
            #file_suffix = f"{start_date.replace('-', '')}"  # 日付の区切りを取り除く
            #csv_file=f"{file_suffix}_{csv_file}"
            #df_data = pd.read_csv(file_path, sep=',')
            df_data2 = pd.read_excel(file_path)
            #df_score2 = df_score[["GhostID", "BirthDate"]]
            #df_data2 = pd.merge(df_data, df_score2, how='left', left_on='GhostID', right_on='GhostID')
            print(df_data2)


            # 全ての列名を取得
            column_names = df_data2.columns

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

            df = pd.DataFrame(df_data2)

            # 'id', 'A', 'B'ごとにデータをグループ化
            grouped = df.groupby(['GhostID', 'BirthDate', 'Event'])
            # 各グループに対して処理を行う
            new_df = pd.DataFrame()

            for name, group in grouped:
                # days列を昇順に並べ替える
                group = group.sort_values(unit_str).reset_index(drop=True)
                
                # 飛んだdaysの値を見つける
                all_days = set(range(group[unit_str].min(), group[unit_str].max() + 1))
                existing_days = set(group[unit_str])
                missing_days = sorted(all_days - existing_days)
                
                # 飛び値がある場合の処理
                if missing_days:
                    # 既存の最大DateからDate列を作成
                    last_date = pd.to_datetime(group[eventdate_str].max())

                    # 足りないdaysの行を作成して追加
                    for i, day in enumerate(missing_days):
                        new_row = {
                                eventdate_str: last_date + pd.Timedelta(days=i+1),  # 日付を1日ずつ増やす
                                unit_str: day, 
                                sum_str: 0, 
                                'GhostID': name[0], 
                                'BirthDate': name[1], 
                                'Event': name[2]
                            }
                        group = pd.concat([group, pd.DataFrame([new_row])], ignore_index=True)
                
                # 再度days列でソート
                group = group.sort_values(unit_str).reset_index(drop=True)
                
                # 結果を新しいデータフレームに追加
                new_df = pd.concat([new_df, group], ignore_index=True)

            print(new_df)
            new_df.to_excel("{}/fft_{}.xlsx".format(out_folder, csv_file))

        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

