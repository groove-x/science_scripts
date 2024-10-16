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
        self.root.title("ui_06_x_days_graph_birth")
        self.input_folder = ""
        self.output_folder = ""

        # ウィンドウサイズの設定
        self.root.geometry("400x400")

        # フォルダ選択ボタン
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 01_data_for_FFT", command=self.select_input_folder)
        self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder： 06_statistic_data_for_FFT", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        self.input_folder_label.pack(pady=5)

        self.output_folder_label = tk.Label(root, text="Output Folder: Not selected")
        self.output_folder_label.pack(pady=5)

        # ファイル選択ボタン
        self.select_file1_btn = tk.Button(root, text="Select Input File：start_date", command=self.select_input_file1)
        self.select_file1_btn.pack(pady=10)

        # ファイルパス表示ラベル
        self.input_file1_label = tk.Label(root, text="Input File 1: Not selected")
        self.input_file1_label.pack(pady=5)
        
        # item入力
        self.item_label = tk.Label(root, text="Enter Items:")
        self.item_label.pack(pady=5)

        self.item_entry = tk.Entry(root)
        self.item_entry.insert(0, "all")  # 初期値を設定
        self.item_entry.pack(pady=5)

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

    def select_input_file1(self):
        self.input_file1 = filedialog.askopenfilename(title="Select the first file to analyze", filetypes=(("xlsx files", "*.xlsx"), ("All files", "*.*")))
        if self.input_file1:
            self.input_file1_label.config(text=f"Input File 1: {os.path.basename(self.input_file1)}")
        else:
            self.input_file1_label.config(text="Input File 1: Not selected")

    def run_analysis(self):
        if not self.input_folder or not self.output_folder or not self.input_file1:
            #messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            print("Warning", "Please select both input and output folders before running the analysis.")
            return

        items = self.item_entry.get()
        if not items:
            #messagebox.showwarning("Warning", "Please enter a valid start date.")
            print("Warning", "Please enter a valid start date.")
            return

        # ここで実際の解析処理を実行します
        file_score = "GhostID-Birthday-score"

        df_score = pd.read_csv("{}.csv".format(file_score), sep=',')
        print(df_score)
        df_score['BirthDate'] = pd.to_datetime(df_score['BirthDate']).dt.date

        #diary_csv_path = './03_data_monthly'  # 相対パスを指定
        diary_csv_path = self.input_folder

        # フォルダ内の全ファイル名を取得
        diary_csv_list = os.listdir(diary_csv_path)
        diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]
        #print(diary_csv_files)

        #d_foldar = "08_相関係数_birth_after202102"
        out_folder = self.output_folder

        df_start_date = pd.read_excel(self.input_file1)
        df_start_date['start_date'] = pd.to_datetime(df_start_date['start_date'])
        print(df_start_date)

        df_adf_data = pd.DataFrame()
        df_exp_data = pd.DataFrame()
        df_wilcoxon_data = pd.DataFrame()

        if "all" in items:
            cal_files = diary_csv_files
        else:
            # 2. カンマで区切られた文字列を抽出
            item_list = items.split(',')  # カンマで分割
             # 指定された項目が含まれるファイル名を抽出
            cal_files = [file for file in diary_csv_files if any(item in file for item in item_list)]
        print(cal_files)

        for csv_file in cal_files:
        #for csv_file in diary_csv_files:
            #if "STROKE" in csv_file:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            df_data2 = pd.read_csv(file_path, sep=',')
            #df_data2 = pd.read_excel(file_path)
            #df_score2 = df_score[["GhostID", "BirthDate"]]
            #df_data2 = pd.merge(df_data, df_score2, how='left', left_on='GhostID', right_on='GhostID')
            print(df_data2)
            df_data2['Timestamp'] = pd.to_datetime(df_data2['Timestamp'])
            df_data2 = pd.merge(df_data2, df_score, how='left', left_on='GhostID', right_on='GhostID')
            df_data2['BirthDate'] = pd.to_datetime(df_data2['BirthDate'])
            df_data2['EventDate'] = df_data2['Timestamp'].dt.date
            df_data2['usedays'] = (pd.to_datetime(df_data2['EventDate']) - pd.to_datetime(df_data2['BirthDate'])).dt.days
            #unit_str = 'usedays'
            #sum_str = 'sum'
            csv_event = df_data2['Event'].unique()
            csv_event_str = csv_event[0]
            print(csv_event_str)

            # A列が 'apple' と一致する行のB列の数値を取り出す
            start_date = df_start_date.loc[df_start_date['Event'] == csv_event_str, 'start_date'].iloc[0]
            #result_date = pd.to_datetime(df_.loc[df['A'] == target_string, 'B'].iloc[0], format='%Y%m%d')
            print(start_date)
            csv_file=csv_file[:-5]
            #start_date_formatted = start_date.strftime('%Y%m%d')
            file_suffix = start_date.strftime('%Y%m%d')
            #file_suffix = f"{start_date.replace('-', '')}"  # 日付の区切りを取り除く
            csv_file=f"{file_suffix}_{csv_file}"

            df_data = df_data2[df_data2['BirthDate'] > start_date]

            # GhostIDの出現回数をカウント
            ghostid_counts = df_data['GhostID'].value_counts().reset_index()
            ghostid_counts.columns = ['GhostID', 'Count']

            # 結果をExcelに保存
            ghostid_counts.to_excel(f"{out_folder}/GhostID_counts_{csv_file}.xlsx", index=False)

            print(df_data)
            #print(df_data['30days_EventDate'])

            #-------------

            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_d = df_data.groupby('Timestamp').agg({
                'sum': ['count', 'mean', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
                #'sum': ['Timestamp', 'mean', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            })
            # 列名を変更する
            grouped_d.columns = ['Count', 'Average', '50th Percentile', '25th Percentile', '75th Percentile']

            grouped_d.to_excel("{}/timestamp_{}.xlsx".format(out_folder, csv_file))

            grouped = grouped_d.copy() 

            # プロットを作成
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 左軸に平均値とパーセンタイルをプロット
            ax1.plot(grouped.index, grouped['Average'], marker='o', color='blue', label='Average')
            ax1.plot(grouped.index, grouped['50th Percentile'], marker='o', color='green', label='median')
            ax1.fill_between(grouped.index, grouped['25th Percentile'], grouped['75th Percentile'], color='gray', alpha=0.3, label='25th-75th Percentile')
            ax1.set_xlabel('usedays')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # 右軸にデータ数をプロット
            ax2 = ax1.twinx()
            ax2.plot(grouped.index, grouped['Count'], marker='x', color='red', linestyle='--', label='Count')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')

            # グラフのタイトルを設定
            #plt.title('Usedays Data with Average, Percentiles, and Count')
            plt.title(csv_file)

            # グラフを表示
            plt.savefig("{}/timestamp_{}.png".format(out_folder, csv_file))
            plt.close('all')

            #-------------
            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_d = df_data.groupby('usedays').agg({
                'sum': ['count', 'mean', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            })
            # 列名を変更する
            grouped_d.columns = ['Count', 'Average', '50th Percentile', '25th Percentile', '75th Percentile']

            grouped_d.to_excel("{}/usedays_{}.xlsx".format(out_folder, csv_file))

            grouped = grouped_d.copy() 
            print(grouped)

            # プロットを作成
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 左軸に平均値とパーセンタイルをプロット
            ax1.plot(grouped.index, grouped['Average'], marker='o', color='blue', label='Average')
            ax1.plot(grouped.index, grouped['50th Percentile'], marker='o', color='green', label='median')
            ax1.fill_between(grouped.index, grouped['25th Percentile'], grouped['75th Percentile'], color='gray', alpha=0.3, label='25th-75th Percentile')
            ax1.set_xlabel('usedays')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # 右軸にデータ数をプロット
            ax2 = ax1.twinx()
            ax2.plot(grouped.index, grouped['Count'], marker='x', color='red', linestyle='--', label='Count')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')

            # グラフのタイトルを設定
            #plt.title('Usedays Data with Average, Percentiles, and Count')
            plt.title(csv_file)

            # グラフを表示
            plt.savefig("{}/usedays_{}.png".format(out_folder, csv_file))
            plt.close('all')

            # ====== ウィルコクソン符号付き順位検定
            u_event = df_data['Event'].unique()
            # A列でソート
            df_sorted = df_data.sort_values(by='Timestamp')

            # A列でグループ化し、B列の値とC列の文字列をリストとして取り出す
            #grouped = df_sorted.groupby(unit_str).apply(lambda x: pd.Series({'B': x[sum_str].tolist(), 'C': x['GhostID'].tolist()}))
            grouped = df_sorted.groupby('Timestamp').apply(lambda x: pd.Series({'B': x['sum'].tolist(), 'C': x['GhostID'].tolist()})).reset_index()

            # ソートされたA列の値
            #a_values = sorted(grouped.keys())
            # ソートされたA列の値
            a_values = sorted(grouped['Timestamp'].unique())
            # ウィルコクソンの符号付き順位検定の結果を格納するリスト
            results = []

            print(grouped)

            # ペアの作成と検定
            for i in range(len(a_values) - 1):
                a_current = a_values[i]
                a_next = a_values[i + 1]
                

                b_current = grouped.loc[grouped['Timestamp'] == a_current, 'B'].values[0]
                c_current = grouped.loc[grouped['Timestamp'] == a_current, 'C'].values[0]
                
                b_next = grouped.loc[grouped['Timestamp'] == a_next, 'B'].values[0]
                c_next = grouped.loc[grouped['Timestamp'] == a_next, 'C'].values[0]
                #b_current = grouped.loc[a_current, 'B']
                #c_current = grouped.loc[a_current, 'C']
                
                #b_next = grouped.loc[a_next, 'B']
                #c_next = grouped.loc[a_next, 'C']
                
                # C列の文字列が一致する条件
                common_strings = set(c_current) & set(c_next)

                # 一致する C 列の文字列に基づいて、A 列と B 列の値を取り出す
                matching_b_current = [b for b, c in zip(b_current, c_current) if c in common_strings]
                matching_b_next = [b for b, c in zip(b_next, c_next) if c in common_strings]

                if len(matching_b_current) > 1 and len(matching_b_next) > 1:
                    if len(set(matching_b_current)) == 1 and len(set(matching_b_next)) == 1:
                        # すべての要素が同じ場合
                        results.append({
                            'Event': u_event[0],
                            'A_current': a_current,
                            'A_next': a_next,
                            'Mean_B_current': np.mean(matching_b_current) if matching_b_current else None,
                            'Mean_B_next': np.mean(matching_b_next) if matching_b_next else None,
                            'Statistic': None,
                            'P_value': None,
                            'n': 'nan',
                            'Effect_size_r': None,
                            'Reason': 'All B values are identical.'
                        })
                    else:
                        # ウィルコクソンの符号付き順位検定
                        try:
                            stat, p_value = wilcoxon(matching_b_current, matching_b_next)
                            n = len(matching_b_current)
                            effect_size = stat / (n * (n + 1) / 2)
                            
                            results.append({
                                'Event': u_event[0],
                                'A_current': a_current,
                                'A_next': a_next,
                                'Mean_B_current': np.mean(matching_b_current) if matching_b_current else None,
                                'Mean_B_next': np.mean(matching_b_next) if matching_b_next else None,
                                'Statistic': stat,
                                'P_value': p_value,
                                'n': n,
                                'Effect_size_r': effect_size
                            })
                        except ValueError as e:
                            results.append({
                                'Event': u_event[0],
                                'A_current': a_current,
                                'A_next': a_next,
                                'Mean_B_current': np.mean(matching_b_current) if matching_b_current else None,
                                'Mean_B_next': np.mean(matching_b_next) if matching_b_next else None,
                                'Statistic': None,
                                'P_value': None,
                                'n': 'nan',
                                'Effect_size_r': None,
                                'Reason': str(e)
                            })
                else:
                    # ペアの長さが一致しない場合、またはC列の文字列が一致しない場合
                    results.append({
                        'Event': u_event[0],
                        'A_next': a_next,
                        'Mean_B_current': None,
                        'Mean_B_next': None,
                        'Statistic': None,
                        'P_value': None,
                        'Effect_size_r': None,
                        'n': 'nan',
                        'Reason': 'No matching C values found for both groups.'
                    })


            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)
            # CSVファイルとして保存
            results_df.to_excel('{}/t_wilcoxon_{}.xlsx'.format(out_folder, csv_file), index=False)

            df_wilcoxon_data = pd.concat([df_wilcoxon_data, results_df], axis=0,ignore_index=True)

            #print("結果が 'wilcoxon_results.csv' として保存されました。")

            #==== ADF検定（Augmented Dickey-Fuller test）：
            # 時系列データが定常性を持つかどうかを調べるための統計的手法

            # 結果を格納するリスト
            results = []

            # C列のユニークな文字列ごとにループ
            for filter_string in df_data['GhostID'].unique():
                # C列の文字列が一致する行を取り出す
                df_filtered = df_data[df_data['GhostID'] == filter_string]
                
                # A列の数値が小さい順に並べる
                df_sorted = df_filtered.sort_values(by='Timestamp')
                
                # B列の値
                b_values = df_sorted['sum']
                n = len(b_values)
                
                # ADF検定を実施
                try:
                    result = adfuller(b_values)
                    adf_statistic = result[0]
                    p_value = result[1]
                    critical_values = result[4]
                    # 結果をリストに追加
                    results.append({
                        'Event': u_event[0],
                        'C_value': filter_string,
                        'n': n,
                        'ADF_Statistic': adf_statistic,
                        'P_value': p_value,
                        'Critical_Value_1%': critical_values['1%'],
                        'Critical_Value_5%': critical_values['5%'],
                        'Critical_Value_10%': critical_values['10%']
                    })
                except Exception as e:
                    # エラーが発生した場合、エラーメッセージを結果に追加
                    results.append({
                        'Event': u_event[0],
                        'C_value': filter_string,
                        'n': n,
                        'ADF_Statistic': None,
                        'P_value': None,
                        'Critical_Value_1%': None,
                        'Critical_Value_5%': None,
                        'Critical_Value_10%': None,
                        'Error': str(e)
                    })

            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)

            # エクセルファイルとして保存
            results_df.to_excel('{}/t_adf_test_{}.xlsx'.format(out_folder, csv_file), index=False)
            df_adf_data = pd.concat([df_adf_data, results_df], axis=0,ignore_index=True)

            #===== 減衰率 =====
            # 結果を格納するリスト
            results = []

            # Category列のユニークな文字列ごとにループ
            for filter_string in df_data['GhostID'].unique():
                # Category列の文字列が一致する行を取り出す
                df_filtered = df_data[df_data['GhostID'] == filter_string]
                
                # Time列の値が小さい順に並べる
                df_sorted = df_filtered.sort_values(by='Timestamp')
                
                # Time列とValue列の値を取得
                t_data = df_sorted['Timestamp']
                y_data = df_sorted['Timestamp']
                
                # 初期パラメータの推定
                initial_guess = [max(y_data), 1.0]
                
                try:
                    # フィッティング
                    params, covariance = curve_fit(exp_decreasing, t_data, y_data, p0=initial_guess)
                    a, b = params
                    
                    # 半減期の計算
                    half_life = np.log(2) / b
                    
                    # 結果をリストに追加
                    results.append({
                        'Event': u_event[0],
                        'Category': filter_string,
                        'a (Amplitude)': a,
                        'b (Decay Rate)': b,
                        'Half-life': half_life
                    })
                except Exception as e:
                    # エラーが発生した場合、エラーメッセージを結果に追加
                    results.append({
                        'Event': u_event[0],
                        'Category': filter_string,
                        'a (Amplitude)': None,
                        'b (Decay Rate)': None,
                        'Half-life': None,
                        'Error': str(e)
                    })

            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)

            # エクセルファイルとして保存
            results_df.to_excel('{}/t_exp_fitting_{}.xlsx'.format(out_folder, csv_file), index=False)
            df_exp_data = pd.concat([df_exp_data, results_df], axis=0,ignore_index=True)

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        df_wilcoxon_data.to_csv('{}/all_t_willcoxon.csv'.format(out_folder), index=False)
        df_adf_data.to_csv('{}/all_t_adf_test.csv'.format(out_folder), index=False)
        df_exp_data.to_csv('{}/all_t_exp_fitting.csv'.format(out_folder), index=False)
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

