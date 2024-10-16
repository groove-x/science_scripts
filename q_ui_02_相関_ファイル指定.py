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
        #self.select_input_btn = tk.Button(root, text="Select Input Folder: 00_data_base", command=self.select_input_folder)
        #self.select_input_btn.pack(pady=10)

        self.select_output_btn = tk.Button(root, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_btn.pack(pady=10)

        # フォルダパス表示ラベル
        self.input_file_label = tk.Label(root, text="Input File: Not selected")
        self.input_file_label.pack(pady=5)
        
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
        
        # ここで実際の解析処理を実行します
        file_data = self.input_file

        df_data = pd.read_csv(file_data, sep=',')
        print(df_data)
        input_file_name = os.path.basename(self.input_file)  # ファイル名だけを取得
        input_file_name = input_file_name[:-4]  # 後ろから4文字を抜く

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

        # 列名に 'columns' のいずれかが含まれているものだけを抽出
        filtered_columns = df_data.columns[df_data.columns.str.contains('|'.join(events))]
        
        columns =["相関係数", "p値", "平均", "中央値", "最大", "最小", "標準偏差", 
                  "正規性：シャピロ・ウィルク検定", "正規性：コルゴモロフ・スミルノフ検定"
                  ]

        df_result = pd.DataFrame(index=filtered_columns, columns=columns)
        #df_data.fillna(0, inplace=True)
        set_score = "score"
        print(df_data)
        for event in filtered_columns:
            print(event)

            each_results = []

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

        df_result.to_excel("{}/{}_相関.xlsx".format(out_folder, input_file_name))
        print(df_result)

        '''
        r_events = ['duration_HUGGED', 'duration_STROKE',
                  "HUGGED", "STROKE", "CALL_NAME", "TOUCH_NOSE", "RELAX", "CHANGE_CLOTHES",
                  "CARRIED_TO_NEST", "LIFTED_UP", "WELCOME_and_WELCOME_GREAT", 
                  "GOOD_MORNING_and_GOOD_MORNING_NEAR_WAKE_TIME",
                  "GOOD_NIGHT_and_GOOD_NIGHT_NEAR_BEDTIME"
                  ]
        '''

        # StandardScalerのインスタンスを作成
        scaler = StandardScaler()

        filtered_events = filtered_columns
        df_merged = df_data

        # eventsにある列を標準化
        df_merged[filtered_events] = scaler.fit_transform(df_merged[filtered_events])
        df_merged['score'] = scaler.fit_transform(df_merged[['score']].values).flatten()
        df_merged.to_csv("{}/{}_標準化.csv".format(out_folder, input_file_name), index=False, encoding='utf-8-sig')  # index=False により、インデックスはCSVに含まれない

        # 説明変数と目的変数の設定
        X = df_merged[filtered_events]  # events列を説明変数として選択
        Y = df_merged['score']     # 目的変数

        # 定数項を加える（切片のため）
        X = sm.add_constant(X)

        # 重回帰モデルの作成
        model = sm.OLS(Y, X)
        results = model.fit()
        # 結果の表示
        print(results.summary())

        # 結果をテキストファイルに出力
        with open("{}/{}_重回帰分析.txt".format(out_folder, input_file_name), 'w') as f:
            f.write(results.summary().as_text())

        #messagebox.showinfo("Info", "Analysis completed and results are saved.")
        print("Analysis completed and results are saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

