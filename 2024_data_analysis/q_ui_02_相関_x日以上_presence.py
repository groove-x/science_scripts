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

class AnalysisApp:
    def __init__(self, root):
        self.root = root
        # プログラムのファイル名を取得して、タイトルに設定
        file_name = os.path.basename(__file__)  # ファイルのフルパスからファイル名を取得
        self.root.title(file_name)  # タイトルをファイル名に設定
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x500")

        # ファイル選択ボタン
        self.select_input_file_btn = tk.Button(root, text="Select 28days_sum_EventDate File", command=self.select_input_file)
        self.select_input_file_btn.pack(pady=10)

        # 日数
        self.filter_label_d = tk.Label(root, text="Enter days:")
        self.filter_label_d.pack(pady=5)

        self.filter_entry_d = tk.Entry(root)
        self.filter_entry_d.pack(pady=5)

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
        
        #self.input_file_d_label = tk.Label(root, text="Input Event File: Not selected")
        #self.input_file_d_label.pack(pady=5)

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
        cal_days = self.filter_entry_d.get().strip()
        # cal_daysをintに変換する
        cal_days = int(cal_days)  # もしくはfloat(cal_days)

        # 入力された文字列を取得
        #filter_text = self.filter_entry.get().strip()

        #if not filter_text:
            #messagebox.showwarning("Warning", "Please enter a filter text.")
            #return

        # ここで実際の解析処理を実行します
        file_score = self.input_file

        df_score = pd.read_csv(file_score, sep=',')
        print(df_score)
        df_score = df_score[df_score['count'] >= cal_days]
        #df_score["q_Date"] = pd.to_datetime(df_score["q_Date"]).dt.date
        df_score = df_score.rename(columns={'count': 'EvantDays'})

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder
        # 最下層フォルダ名を取得
        input_deepest_folder = os.path.basename(diary_csv_path)

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]
        # 文字列がファイル名の先頭にあるファイルをフィルタリング
        #filtered_csv_files = [file for file in diary_csv_files if file.startswith(filter_text)]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder
        
        #df_merged = df_score.copy()

        for csv_file in diary_csv_files:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            #df_data = pd.read_csv(file_path)
            df_data = pd.read_csv(file_path, sep=',')

            # ファイル名に含まれているイベント名を取り出す
            df_selected = df_data[['q_Date+GhostID', 'count', 'score']]  # 'A' 列と 'Data' 列を選択

            df_selected = df_selected.fillna(0)

            # DataFrameAのA列に存在する値のみを保持する
            df_selected = df_selected[df_selected['q_Date+GhostID'].isin(df_score['q_Date+GhostID'])]
        
            df_selected.to_csv("{}/{}_{}_merged_{}.csv".format(out_folder, input_deepest_folder, cal_days, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

            columns =["相関係数", "p値", "平均", "中央値", "最大", "最小", "標準偏差", 
                    "正規性：シャピロ・ウィルク検定", "正規性：コルゴモロフ・スミルノフ検定"
                    ]

            df_result = pd.DataFrame(columns=columns)
            #df_data.fillna(0, inplace=True)
            df_data = df_selected
            set_score = "score"
            print(df_data)

            event = 'count'

            # 相関係数とp値の計算
            #print(df_data[event])
            print(df_result)
            # 欠損値を0で置き換える
            #スピアマンで計算しなおす
            df_data = df_data.dropna(subset=['score', event])

            corr, p_value = spearmanr(df_data[set_score], df_data[event])
            #spearman_corr = df_data[set_score].corr(df_data[event], method='spearman')
            #corr, p_value = pearsonr(df_data[set_score], df_data[event])
            #corr= df_data[[set_score,event]].corr().iloc[0,1]
            print(corr)
            #df_result.at[1, "相関係数"] = 1
            df_result.loc[event, "相関係数"] = corr
            #df_result.at[event, "相関係数"] = corr
            #p_value= df_data[[set_score,event]].corr().iloc[0,1]
            df_result.loc[event, "p値"] = p_value
            print(df_data[event])
            print(event)
            print(corr)

            print("確認")
            print(df_data[set_score].unique())
            print(df_data[event].unique())

            print(df_data[set_score].describe())
            print(df_data[event].describe())

            # 平均、中央値、最大値、最小値の計算
            mean= df_data[event].mean()
            median = df_data[event].median()
            max_value= df_data[event].max()
            min_value= df_data[event].min()
            df_result.loc[event, "平均"] = mean
            df_result.loc[event, "中央値"] = median
            df_result.loc[event, "最大"] = max_value
            df_result.loc[event, "最小"] = min_value

            # 偏差の計算
            std_deviation= df_data[event].std()
            df_result.loc[event, "標準偏差"] = std_deviation

            # 正規分布の判定
            statistic, p_value = shapiro(df_data[event])
            df_result.loc[event, "正規性：シャピロ・ウィルク検定"] = p_value
            statistic, p_value = kstest(df_data[event], 'norm')
            df_result.loc[event, "正規性：コルゴモロフ・スミルノフ検定"] = p_value

            df_result.to_excel("{}/{}_{}_相関_{}.xlsx".format(out_folder, input_deepest_folder, cal_days, csv_file))
            print(df_result)

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

