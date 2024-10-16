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
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

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

        # フィルタリングに使う文字を入力するテキストフィールド
        #self.filter_label = tk.Label(root, text="Enter filter text:")
        #self.filter_label.pack(pady=5)

        #self.filter_entry = tk.Entry(root)
        #self.filter_entry.pack(pady=5)

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
        #filter_text = self.filter_entry.get().strip()

        #if not filter_text:
            #messagebox.showwarning("Warning", "Please enter a filter text.")
            #return

        # ここで実際の解析処理を実行します
        file_score = self.input_file

        df_score = pd.read_csv(file_score, sep=',')
        print(df_score)
        df_score["q_Date"] = pd.to_datetime(df_score["q_Date"]).dt.date

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder
        # 最下層フォルダ名を取得
        input_deepest_folder = os.path.basename(diary_csv_path)

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('_merged.csv')]
        # 文字列がファイル名の先頭にあるファイルをフィルタリング
        #filtered_csv_files = [file for file in diary_csv_files if file.startswith(filter_text)]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder

        events = ['duration_HUGGED', 'duration_STROKE',
                  "HUGGED", "STROKE", "CALL_NAME", "TOUCH_NOSE", "RELAX", "CHANGE_CLOTHES",
                  "CARRIED_TO_NEST", "LIFTED_UP", "WELCOME_and_WELCOME_GREAT", 
                  "GOOD_MORNING_and_GOOD_MORNING_NEAR_WAKE_TIME",
                  "GOOD_NIGHT_and_GOOD_NIGHT_NEAR_BEDTIME", "HELLO",
                  "SING_FAVORITE", "SING", "REMEMBER_FAVORITE",
                  'OUCH', 'HELP', 'TAKE_PICTURE'
                  ]
        
        df_merged = df_score.copy()

        f_names = ['01_FRQ', '02_HBT', 
                    'HBT_sum_sleep', 'HBT_sum_wake','FRQ_sum_sleep', 'FRQ_sum_wake',
                    'HBT_hugged', 'H-FRQ_sum_hugged', 'HBT_sum_hugged_before', 'HBT_sum_hugged_later',
                    'FRQ_sum_hugged_before', 'FRQ_sum_hugged_later_1min'
                    ]

        for csv_file in diary_csv_files:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            #df_data = pd.read_csv(file_path)
            df_data = pd.read_csv(file_path, sep=',')

             # df_data の columns にある列だけを残す
            matched_events = [event for event in events if event in df_data.columns]
            df_selected = df_data[['q_Date+GhostID'] + matched_events]  # 'A' 列と 'Data' 列を選択
            print(df_selected)
            # 含まれている文字列を取り出す
            f_name = [f for f in f_names if f in csv_file]
            print(f_name)

            # matched_events に含まれる列名の後ろに "_A" を追加する
            df_selected = df_selected.rename(columns={col: col + '_' + f_name[0] for col in matched_events})
            print(df_selected)

            df_merged = pd.merge(df_merged, df_selected, on='q_Date+GhostID', how='left')
            print(df_merged)
        
        df_merged.to_csv("{}/merged.csv".format(out_folder), index=False)  # index=False により、インデックスはCSVに含まれない

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

