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
        self.root.title("ui_08_相関係数_limit_birth")
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
        d_foldar = self.output_folder

        for csv_file in diary_csv_files:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            csv_file=csv_file[:-5]
            #df_data = pd.read_csv(file_path, sep=',')
            df_data2 = pd.read_excel(file_path)
            #df_score2 = df_score[["GhostID", "BirthDate"]]
            #df_data2 = pd.merge(df_data, df_score2, how='left', left_on='GhostID', right_on='GhostID')
            print(df_data2)

            df_data2['30days_EventDate'] = pd.to_datetime(df_data2['30days_EventDate'])
            df_data2['BirthDate'] = pd.to_datetime(df_data2['BirthDate'])
            #start_date = '2021-02-01'
            start_date = '2022-11-01'
            df_data = df_data2[df_data2['BirthDate'] > start_date]


            #-----------
            # 欠損値を無視する
            df_data = df_data.dropna(subset=['30days_sum'])

            print(df_data)
            #print(df_data['30days_EventDate'])


            correlation_results = df_data.groupby('30days_unit').apply(spearman_corr_p, '30days_sum').reset_index()
            print(correlation_results)

            grouped = correlation_results.copy() 

            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_d = df_data.groupby('30days_unit').agg({
                '30days_sum': ['count', 
                            'mean', 
                            'std',  # 標準偏差を追加
                            lambda x: x.quantile(0.5), 
                            lambda x: x.quantile(0.25), 
                            lambda x: x.quantile(0.75)
                            ]
            })
            # 列名を変更する
            grouped_d.columns = ['Count', 'Average', 'STD', '50th Percentile', '25th Percentile', '75th Percentile']

            # 複数の列を参照して結合
            grouped_d_r = pd.merge(grouped_d, correlation_results, on=['30days_unit'], how='inner', suffixes=('', '_drop'))

            # 重複した列を削除
            grouped_d_r.drop([col for col in grouped_d_r.columns if 'drop' in col], axis=1, inplace=True)

            grouped_d_r.to_excel("{}/useday_count_corr_{}.xlsx".format(d_foldar ,csv_file))

            # プロットを作成
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 左軸に平均値とパーセンタイルをプロット
            ax1.plot(grouped['30days_unit'], grouped['Spearman_Correlation'], marker='o', color='blue', label='corr')
            ax1.set_xlabel('usedays')
            ax1.set_ylabel('corr')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # 右軸にデータ数をプロット
            ax2 = ax1.twinx()
            ax2.plot(grouped['30days_unit'], grouped['p_value'], marker='x', color='red', linestyle='--', label='p_value')
            ax2.set_ylabel('p_value')
            ax2.legend(loc='upper right')

            # グラフのタイトルを設定
            plt.title('Usedays Data with Average, Percentiles, and Count')

            # グラフを表示
            plt.savefig("{}/usedays_count_{}.png".format(d_foldar, csv_file))
            plt.close('all')

            #-----------
            # 欠損値を無視する
            df_data = df_data.dropna(subset=['change_360'])

            print(df_data)
            #print(df_data['30days_EventDate'])


            correlation_results = df_data.groupby('30days_unit').apply(spearman_corr_p, 'change_360').reset_index()
            print(correlation_results)

            grouped = correlation_results.copy() 

            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_d = df_data.groupby('30days_unit').agg({
                'change_360': ['count', 
                            'mean', 
                            'std',  # 標準偏差を追加
                            lambda x: x.quantile(0.5), 
                            lambda x: x.quantile(0.25), 
                            lambda x: x.quantile(0.75)
                            ]
            })
            # 列名を変更する
            grouped_d.columns = ['Count', 'Average', 'STD', '50th Percentile', '25th Percentile', '75th Percentile']

            # 複数の列を参照して結合
            grouped_d_r = pd.merge(grouped_d, correlation_results, on=['30days_unit'], how='inner', suffixes=('', '_drop'))

            # 重複した列を削除
            grouped_d_r.drop([col for col in grouped_d_r.columns if 'drop' in col], axis=1, inplace=True)

            grouped_d_r.to_excel("{}/useday_360_corr_{}.xlsx".format(d_foldar ,csv_file))

            # プロットを作成
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 左軸に平均値とパーセンタイルをプロット
            ax1.plot(grouped['30days_unit'], grouped['Spearman_Correlation'], marker='o', color='blue', label='corr')
            ax1.set_xlabel('usedays')
            ax1.set_ylabel('corr')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # 右軸にデータ数をプロット
            ax2 = ax1.twinx()
            ax2.plot(grouped['30days_unit'], grouped['p_value'], marker='x', color='red', linestyle='--', label='p_value')
            ax2.set_ylabel('p_value')
            ax2.legend(loc='upper right')

            # グラフのタイトルを設定
            plt.title('Usedays Data with Average, Percentiles, and Count')

            # グラフを表示
            plt.savefig("{}/usedays_360_{}.png".format(d_foldar, csv_file))
            plt.close('all')

            #---------

            df_data = df_data2[df_data2['BirthDate'] > start_date]

            # 欠損値を無視する
            df_data = df_data.dropna(subset=['change_0'])

            print(df_data)
            #print(df_data['30days_EventDate'])
            if not df_data.empty:
                correlation_results = df_data.groupby('30days_unit').apply(spearman_corr_p, 'change_0').reset_index()
                print(correlation_results)

                grouped = correlation_results.copy() 

                # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
                grouped_d = df_data.groupby('30days_unit').agg({
                    'change_0': ['count', 
                                'mean', 
                                'std',  # 標準偏差を追加
                                lambda x: x.quantile(0.5), 
                                lambda x: x.quantile(0.25), 
                                lambda x: x.quantile(0.75)
                                ]
                })
                # 列名を変更する
                grouped_d.columns = ['Count', 'Average', 'STD', '50th Percentile', '25th Percentile', '75th Percentile']

                # 複数の列を参照して結合
                grouped_d_r = pd.merge(grouped_d, correlation_results, on=['30days_unit'], how='inner', suffixes=('', '_drop'))

                # 重複した列を削除
                grouped_d_r.drop([col for col in grouped_d_r.columns if 'drop' in col], axis=1, inplace=True)

                grouped_d_r.to_excel("{}/useday_0_corr_{}.xlsx".format(d_foldar ,csv_file))

                # プロットを作成
                fig, ax1 = plt.subplots(figsize=(10, 6))

                # 左軸に平均値とパーセンタイルをプロット
                ax1.plot(grouped['30days_unit'], grouped['Spearman_Correlation'], marker='o', color='blue', label='corr')
                ax1.set_xlabel('usedays')
                ax1.set_ylabel('corr')
                ax1.legend(loc='upper left')
                ax1.grid(True)

                # 右軸にデータ数をプロット
                ax2 = ax1.twinx()
                ax2.plot(grouped['30days_unit'], grouped['p_value'], marker='x', color='red', linestyle='--', label='p_value')
                ax2.set_ylabel('p_value')
                ax2.legend(loc='upper right')

                # グラフのタイトルを設定
                plt.title('Usedays Data with Average, Percentiles, and Count')

                # グラフを表示
                plt.savefig("{}/usedays_0_{}.png".format(d_foldar, csv_file))
                plt.close('all')

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()
