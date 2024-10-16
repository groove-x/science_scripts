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
        self.root.geometry("400x500")

        # ファイル選択ボタン
        self.select_input_file_btn = tk.Button(root, text="Select analysis File", command=self.select_input_file)
        self.select_input_file_btn.pack(pady=10)

        # フィルタリングに使う文字を入力するテキストフィールド
        self.filter_label1 = tk.Label(root, text="Enter group Event1 name:")
        self.filter_label1.pack(pady=5)

        self.filter_entry1 = tk.Entry(root)
        self.filter_entry1.pack(pady=5)

        # グループを分ける値
        self.filter_label_g1 = tk.Label(root, text="Enter g1-value:")
        self.filter_label_g1.pack(pady=5)

        self.filter_entry_g1 = tk.Entry(root)
        self.filter_entry_g1.pack(pady=5)

        # フォルダ選択ボタン
        #self.select_input_btn = tk.Button(root, text="Select Input Folder: 00_data_base", command=self.select_input_folder)
        #self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_file_label = tk.Label(root, text="Input File: Not selected")
        self.input_file_label.pack(pady=5)
        
        #self.input_file_d_label = tk.Label(root, text="Input Event File: Not selected")
        #self.input_file_d_label.pack(pady=5)

        #self.input_folder_label = tk.Label(root, text="Input Folder: Not selected")
        #self.input_folder_label.pack(pady=5)

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
    '''
    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory(title="Select the folder to analyze")
        if self.input_folder:
            self.input_folder_label.config(text=f"Input Folder: {os.path.basename(self.input_folder)}")
        else:
            self.input_folder_label.config(text="Input Folder: Not selected")
    '''
    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory(title="Select the folder to save results")
        if self.output_folder:
            self.output_folder_label.config(text=f"Output Folder: {os.path.basename(self.output_folder)}")
        else:
            self.output_folder_label.config(text="Output Folder: Not selected")

    def run_analysis(self):
        if not self.output_folder or not self.input_file:
            messagebox.showwarning("Warning", "Please select both input and output folders before running the analysis.")
            return

        # 入力された文字列を取得
        filter_event_name1 = self.filter_entry1.get().strip()
        g1_value = self.filter_entry_g1.get().strip()
        # floatに変換する
        g1_value = float(g1_value)  # もしくはfloat(cal_days)

        # ここで実際の解析処理を実行します
        file_score = self.input_file

        df_score = pd.read_csv(file_score, sep=',')
        print(df_score)

        out_folder = self.output_folder
        
        #df_selected = df_score[['score']+filter_text]  # 'A' 列と 'Data' 列を選択
        events = ['score', 'duration_HUGGED', 'duration_STROKE',
        "HUGGED", "STROKE", "CALL_NAME", "TOUCH_NOSE", "RELAX", "CHANGE_CLOTHES",
        "CARRIED_TO_NEST", "LIFTED_UP", "WELCOME_and_WELCOME_GREAT", 
        "GOOD_MORNING_and_GOOD_MORNING_NEAR_WAKE_TIME",
        "GOOD_NIGHT_and_GOOD_NIGHT_NEAR_BEDTIME", "HELLO",
        "SING_FAVORITE", "SING", "REMEMBER_FAVORITE",
        'OUCH', 'HELP', 'TAKE_PICTURE'
        ]


        columns =["p_H-L", "effect_r_H-L",
                "H_n", "L_n", 
                "H_median", "L_median", 
                "H_mean", "L_mean", 
                "H_std", "L_std"
                  ]

        print(df_score)
        # DataFrame に存在する events 列だけをフィルタリング
        #filtered_events = [col for col in events if col in df_score.columns]
        # 列名に 'columns' のいずれかが含まれているものだけを抽出
        filtered_events = df_score.columns[df_score.columns.str.contains('|'.join(events))]

        df_result = pd.DataFrame(index=filtered_events, columns=columns)

        #df_result = pd.DataFrame(index=['score_{}'.format(filter_text)], columns=[columns])

        # 4つのグループに分ける
        df_score['group'] = 'Other'  # 初期化
        df_score.loc[(df_score[filter_event_name1] >= g1_value, 'group')] = 'H'
        df_score.loc[(df_score[filter_event_name1] <  g1_value, 'group')] = 'L'

        for event in filtered_events:
            # Kruskal-Wallis検定の実施# グループごとのスコアをリストに格納
            group_H = df_score[df_score['group'] == 'H'][event]
            group_L = df_score[df_score['group'] == 'L'][event]

            # グループごとのデータを抽出
            group_data = {
                'H': group_H,
                'L': group_L,
            }
            groups = ['H', 'L']

            for group in groups:
                group_i = group_data[group]  # グループに該当するスコアを取得
                df_result.loc[event, f"{group}_median"] = np.median(group_i)
                df_result.loc[event, f"{group}_n"] = len(group_i)
                df_result.loc[event, f"{group}_mean"] = np.mean(group_i)
                df_result.loc[event, f"{group}_std"] = np.std(group_i, ddof=1)

            #num_comparisons = len(groups) * (len(groups) - 1) // 2  # ペアごとの比較回数
            num_comparisons = 1
            comparison_results = []

            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    group1 = df_score[df_score['group'] == groups[i]][event]
                    group2 = df_score[df_score['group'] == groups[j]][event]
                    mannwhitney_result = stats.mannwhitneyu(group1, group2)
                    corrected_pvalue = mannwhitney_result.pvalue * num_comparisons  # Bonferroni修正
                    group1_median = np.median(group1)
                    group2_median = np.median(group2)
                    #cohen_r = (group1_median - group2_median) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
                    r = mannwhitney_result.statistic / (len(group1) * len(group2))  # type1とtype2の間の効果量
                    cohen_r = r
                    comparison_results.append((groups[i], groups[j], corrected_pvalue, cohen_r))

                    df_result.loc[event, f"p_{groups[i]}-{groups[j]}"] = corrected_pvalue
                    df_result.loc[event, f"effect_r_{groups[i]}-{groups[j]}"] = cohen_r

            # p値を1に制限
            #corrected_pvalue = min(corrected_pvalue, 1)

            # 効果量 (Cohen's r) の計算
            #U = mannwhitney_result.statistic
            #n1 = len(group_LG)
            #n2 = len(group_HG)
            #cohen_r = U / (n1 * n2)


        print(df_result)
        df_result.to_csv("{}/群間検定_{}{}.csv".format(out_folder, filter_event_name1, g1_value))  # index=False により、インデックスはCSVに含まれない

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

