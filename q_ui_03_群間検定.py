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
        self.filter_label = tk.Label(root, text="Enter Event name:")
        self.filter_label.pack(pady=5)

        self.filter_entry = tk.Entry(root)
        self.filter_entry.pack(pady=5)

        # グループを分ける値
        self.filter_label_g = tk.Label(root, text="Enter g-value:")
        self.filter_label_g.pack(pady=5)

        self.filter_entry_g = tk.Entry(root)
        self.filter_entry_g.pack(pady=5)


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
        filter_text = self.filter_entry.get().strip()
        g_value = self.filter_entry_g.get().strip()
        # floatに変換する
        g_value = float(g_value)  # もしくはfloat(cal_days)

        if not filter_text:
            messagebox.showwarning("Warning", "Please enter a filter text.")
            return

        # ここで実際の解析処理を実行します
        file_score = self.input_file

        df_score = pd.read_csv(file_score, sep=',')
        print(df_score)

        out_folder = self.output_folder
        
        #df_selected = df_score[['score']+filter_text]  # 'A' 列と 'Data' 列を選択

        columns =["p_HG-LG", "effect_r_HG-LG",
                "HG_n", "LG_n", "HG_median", "LG_median",
                "HG_mean", "LG_mean", "HG_std", "LG_std", 
                  ]

        df_result = pd.DataFrame(index=['score_{}'.format(filter_text)], columns=[columns])

        # 閾値より大きいグループと小さいグループに分ける
        group_LG = df_score[df_score[filter_text] < g_value]['score']
        group_HG = df_score[df_score[filter_text] >= g_value]['score']

        df_result.loc['score_{}'.format(filter_text), "LG_median"] = np.median(group_LG)
        df_result.loc['score_{}'.format(filter_text), "HG_median"] = np.median(group_HG)
        df_result.loc['score_{}'.format(filter_text), "LG_n"] = len(group_LG)
        df_result.loc['score_{}'.format(filter_text), "HG_n"] = len(group_HG)
        df_result.loc['score_{}'.format(filter_text), "LG_mean"] = np.mean(group_LG)
        df_result.loc['score_{}'.format(filter_text), "HG_mean"] = np.mean(group_HG)
        df_result.loc['score_{}'.format(filter_text), "LG_std"] = np.std(group_LG, ddof=1)
        df_result.loc['score_{}'.format(filter_text), "HG_std"] = np.std(group_HG, ddof=1)

        #num_comparisons = len(group_LG) * (len(group_HG) - 1) // 2  # ペアごとの比較回数
        num_comparisons = 1
        mannwhitney_result = stats.mannwhitneyu(group_LG, group_HG)
        corrected_pvalue = mannwhitney_result.pvalue * num_comparisons  # Bonferroni修正
        #cohen_r = (group1_median - group2_median) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
        r = mannwhitney_result.statistic / (len(group_LG) * len(group_HG))  # type1とtype2の間の効果量
        cohen_r = r
        df_result.loc['score_{}'.format(filter_text), "p_HG-LG"] = corrected_pvalue
        df_result.loc['score_{}'.format(filter_text), "effect_r_HG-LG"] = cohen_r

        # p値を1に制限
        #corrected_pvalue = min(corrected_pvalue, 1)

        # 効果量 (Cohen's r) の計算
        #U = mannwhitney_result.statistic
        #n1 = len(group_LG)
        #n2 = len(group_HG)
        #cohen_r = U / (n1 * n2)


        print(df_result)
        df_result.to_csv("{}/群間検定_{}_{}.csv".format(out_folder, filter_text, g_value))  # index=False により、インデックスはCSVに含まれない

        print(group_HG)

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

