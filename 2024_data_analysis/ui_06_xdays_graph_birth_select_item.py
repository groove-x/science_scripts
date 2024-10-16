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
        self.select_input_btn = tk.Button(root, text="Select Input Folder: 03_monthly", command=self.select_input_folder)
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
        out_folder = self.output_folder

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

        for csv_file in cal_files:
        #for csv_file in diary_csv_files:
            #if "STROKE" in csv_file:
            print(csv_file)
            file_path = os.path.join(diary_csv_path, csv_file)
            csv_file=csv_file[:-5]
            file_suffix = f"{start_date.replace('-', '')}"  # 日付の区切りを取り除く
            csv_file=f"{file_suffix}_{csv_file}"
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

            print(df_data2[eventdate_str])
            # もし列が既に datetime 型なら、変換をスキップ
            #if not pd.api.types.is_datetime64_any_dtype(df_data2[eventdate_str]):
            df_data2[eventdate_str] = pd.to_datetime(df_data2[eventdate_str])
            #df_data2['every_days_EventDate'] = pd.to_datetime(df_data2['every_days_EventDate'])
            #df_data2['30days_EventDate'] = pd.to_datetime(df_data2['30days_EventDate'])
            df_data2['BirthDate'] = pd.to_datetime(df_data2['BirthDate'])
            #start_date = '2022-11-01'
            #start_date = '2021-02-01'
            df_data = df_data2[df_data2['BirthDate'] > start_date]

            # GhostIDの出現回数をカウント
            ghostid_counts = df_data['GhostID'].value_counts().reset_index()
            ghostid_counts.columns = ['GhostID', 'Count']

            # 結果をExcelに保存
            ghostid_counts.to_excel(f"{out_folder}/GhostID_counts_{csv_file}.xlsx", index=False)

            print(df_data)
            #print(df_data['30days_EventDate'])

            #-------------

            # 月ごとにデータを平均する
            monthly_avg = df_data.resample('M', on=eventdate_str)[sum_str].mean().rename('Average')
            monthly_25 = df_data.resample('M', on=eventdate_str)[sum_str].quantile(0.25).rename('25th Percentile')
            monthly_50 = df_data.resample('M', on=eventdate_str)[sum_str].quantile(0.50).rename('50th Percentile')
            monthly_75 = df_data.resample('M', on=eventdate_str)[sum_str].quantile(0.75).rename('75th Percentile')
            monthly_count = df_data.resample('M', on=eventdate_str)[sum_str].count().rename('Count')

            # これらのデータを一つのDataFrameに結合する
            result = pd.concat([monthly_avg, monthly_50, monthly_25, monthly_75, monthly_count], axis=1)
            print(result)

            result.to_excel("{}/monthly_{}.xlsx".format(out_folder, csv_file))

            # プロットを作成
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # 左軸に平均値とパーセンタイルをプロット
            ax1.plot(result.index, result['Average'], marker='o', color='blue', label='Average')
            ax1.plot(result.index, result['50th Percentile'], marker='o', color='green', label='median')
            ax1.fill_between(result.index, result['25th Percentile'], result['75th Percentile'], color='gray', alpha=0.3, label='25th-75th Percentile')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Value')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # 右軸にデータ数をプロット
            ax2 = ax1.twinx()
            ax2.plot(result.index, result['Count'], marker='x', color='red', linestyle='--', label='Count')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')

            # グラフのタイトルを設定
            plt.title(csv_file)
            #plt.title('Monthly Data with Average, Percentiles, and Count')

            # グラフを表示
            plt.savefig("{}/monthly_{}.png".format(out_folder, csv_file))
            plt.close('all')

            #-------------
            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_d = df_data.groupby(unit_str).agg({
                sum_str: ['count', 'mean', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            })
            # 列名を変更する
            grouped_d.columns = ['Count', 'Average', '50th Percentile', '25th Percentile', '75th Percentile']

            grouped_d.to_excel("{}/usedays_{}.xlsx".format(out_folder, csv_file))

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
            plt.savefig("{}/usedays_{}.png".format(out_folder, csv_file))
            plt.close('all')

            #-------------

            # 'category'列の数値が同じ行をグループ化し、データ数、平均、パーセンタイルの50を計算する
            grouped_c = df_data.groupby(unit_str).agg({
                'change_0': ['count', 'mean', lambda x: x.quantile(0.5), lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
            })
            # 列名を変更する
            grouped_c.columns = ['Count', 'c_Average', 'c_50th Percentile', 'c_25th Percentile', 'c_75th Percentile']

            # 複数の列を参照して結合
            grouped_d_c = pd.merge(grouped_d, grouped_c, on=[unit_str], how='inner', suffixes=('', '_drop'))

            # 重複した列を削除
            grouped_d_c.drop([col for col in grouped_d_c.columns if 'drop' in col], axis=1, inplace=True)

            grouped_d_c.to_excel("{}/usedays_change_0_{}.xlsx".format(out_folder, csv_file))

            grouped = grouped_c.copy() 

            # 列名を変更する
            grouped.columns = ['Count', 'Average', '50th Percentile', '25th Percentile', '75th Percentile']

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
            plt.savefig("{}/usedays_change_0_{}.png".format(out_folder, csv_file))
            plt.close('all')

            # ====== ウィルコクソン符号付き順位検定
            u_event = df_data['Event'].unique()
            # A列でソート
            df_sorted = df_data.sort_values(by=unit_str)

            # A列でグループ化し、B列の値とC列の文字列をリストとして取り出す
            #grouped = df_sorted.groupby(unit_str).apply(lambda x: pd.Series({'B': x[sum_str].tolist(), 'C': x['GhostID'].tolist()}))
            grouped = df_sorted.groupby(unit_str).apply(lambda x: pd.Series({'B': x[sum_str].tolist(), 'C': x['GhostID'].tolist()})).reset_index()

            # ソートされたA列の値
            #a_values = sorted(grouped.keys())
            # ソートされたA列の値
            a_values = sorted(grouped[unit_str].unique())
            # ウィルコクソンの符号付き順位検定の結果を格納するリスト
            results = []

            print(grouped)

            # ペアの作成と検定
            for i in range(len(a_values) - 1):
                a_current = a_values[i]
                a_next = a_values[i + 1]
                

                b_current = grouped.loc[grouped[unit_str] == a_current, 'B'].values[0]
                c_current = grouped.loc[grouped[unit_str] == a_current, 'C'].values[0]
                
                b_next = grouped.loc[grouped[unit_str] == a_next, 'B'].values[0]
                c_next = grouped.loc[grouped[unit_str] == a_next, 'C'].values[0]
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
                df_sorted = df_filtered.sort_values(by=unit_str)
                
                # B列の値
                b_values = df_sorted[sum_str]
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
                df_sorted = df_filtered.sort_values(by=unit_str)
                
                # Time列とValue列の値を取得
                t_data = df_sorted[unit_str]
                y_data = df_sorted[sum_str]
                
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

