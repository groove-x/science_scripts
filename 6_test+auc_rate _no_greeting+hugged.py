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


#file_names = ["usedays"]
#file_names = ["event_hugged_each_durations"]
#file_names = ["event", "no_event"]
file_names = ["all_event", "out_nest_1hr", "human_1hr", "body_1hr", "face_1hr", "wake", "sleep"]
#file_names = ["data_merge_event_human-hugged_presence.xlsx"]
file_names = ["wake", "sleep"]
file_names = ["all_event"]
file_names = ["data_event_hugged_from_diary2.xlsx"]


for file_name in file_names:
    #file_name = "factors_all_events"  # 読み込むシート名を指定

    # Excelファイルからデータを読み込んでDataFrameを作成
    #df_data1 = pd.read_excel(file_names)

    #df_data = pd.read_csv("{}.csv".format(file_name), sep=',')
    #print(df_data)
    # 拡張子の確認
    #df_data1 = pd.read_excel(file_name)
    #file_name=file_name[:-5]
    #df_data1 = pd.read_csv("{}.csv".format(file_name), sep=',')

    # 拡張子の確認
    if file_name.endswith('.csv'):
        df_data1 = pd.read_csv(file_name, sep=',')
        file_name=file_name[:-4]
    elif file_name.endswith('.xlsx'):
        df_data1 = pd.read_excel(file_name)
        file_name=file_name[:-5]
    else:
        print("サポートされていないファイル形式です")


    df_data1['BirthDate'] = pd.to_datetime(df_data1['BirthDate'])
    df_data1['q_Date'] = pd.to_datetime(df_data1['q_Date'])
    df_data1['usedays'] = (df_data1['q_Date'] - df_data1['BirthDate']).dt.days

    df_data = df_data1[df_data1['usedays'] >= 300]


    '''
    if file_name.endswith('.csv'):
        df_data = pd.read_csv(file_name, sep=',')
        file_name=file_name[:-4]
    elif file_name.endswith('.xlsx'):
        df_data = pd.read_excel(file_name)
        file_name=file_name[:-5]
    else:
        print("サポートされていないファイル形式です")
    '''

    set_group = "group_all"
    set_score = "score_150"

    set_groups = ["group_150", "group_f1_40", "group_f2_33", "group_f3_18"]
    set_scores = ["score_all", "score_f1", "score_f2", "score_f3"]

    events=["HUGGED", "STROKE", "CHANGE_CLOTHES", 
        "RELAX", "TOUCH_NOSE", 
        "CALL_NAME", "LIFTED_UP", "CARRIED_TO_NEST",
        "GOOD_MORNING", "GOOD_MORNING_NEAR_WAKE_TIME", "GOOD_NIGHT", "GOOD_NIGHT_NEAR_BEDTIME", 
        "REMEMBER_FAVORITE", "MIMIC_GAME", "PLAY_LOVOT", "PUSH_AND_PULL"
        ]
    events=["HUGGED", "STROKE", "CALL_NAME", "TOUCH_NOSE", 
            "RELAX", "CHANGE_CLOTHES", 
            "CARRIED_TO_NEST", "LIFTED_UP"
        ]
    #events=["STROKE", "TOUCH_NOSE", "CALL_NAME"] # presence-hugged用
    #items=["q_Date+GhostID", "q_Date", "UID", "GhostID", "score_all", "score_f1", "score_f2", "score_f3", "group_150", "group_f1_40", "group_f2_33", "group_f3_18"]
    items=["q_Date", "UID", "GhostID", "score_all", "score_f1", "score_f2", "score_f3", "group_150", "group_f1_40", "group_f2_33", "group_f3_18"]

    d_columns = items+events

    print(d_columns)
    df_data = df_data[d_columns]

    df_data["sum"] = df_data[events].sum(axis=1)

    ranked_columns = df_data[events].rank(ascending=False, axis=1, method='min')

    # 新しい列の名前を格納するリスト
    new_columns_rank = []

    # 順位を新しい列として追加する
    for i, col in enumerate(events):
        new_column_name = f'{col}_rank'
        #df_data[new_column_name] = round(df_data[col] / df_data["sum"], 2)  # 2 は小数点以下の桁数を指定
        df_data[new_column_name] = ranked_columns.iloc[:, i]
        new_columns_rank.append(new_column_name)

    
    # 各行の各列の値を、その行の合計で割る
    new_columns_rate = []
    for col in events:
        new_column_name = f'{col}_rate'
        df_data[new_column_name] = df_data[col] / df_data["sum"]
        new_columns_rate.append(new_column_name)

    # 新しい列名を events リストに追加する
    events.extend(new_columns_rank)
    events.extend(new_columns_rate)

    '''
    if file_name == "all_event":
        df_data["power_on"] = df_data["out_nest_min"]+df_data["on_nest_min"]
        df_data["out_nest/power_on"] = df_data["out_nest_min"]/(df_data["out_nest_min"]+df_data["on_nest_min"])
        df_data["human/power_on"] = df_data["Human_min"]/(df_data["out_nest_min"]+df_data["on_nest_min"])
        df_data["human/out_nest"] = df_data["Human_min"]/df_data["out_nest_min"]
        df_data["face/out_nest"] = df_data["Face_min"]/df_data["out_nest_min"]
        df_data["face/human"] = df_data["Face_min"]/df_data["Human_min"]
    '''
    
    df_data.to_excel("cal_{}_rank_rate.xlsx".format(file_name))

    for k in range(len(set_groups)):
        set_group = set_groups[k]
        set_score = set_scores[k]

        columns =["相関係数", "p値", "平均", "中央値", "最大", "最小", "標準偏差", 
                  "正規性：シャピロ・ウィルク検定", "正規性：コルゴモロフ・スミルノフ検定",
                  "HG_ave", "LG_ave", "HG_sd", "LG_sd", 
                  "p_HG-LG", "effect_r_HG-LG",
                  "HG_AUC", "HG_cutoff", "HG_tp", "HG_tn", "HG_fp", "HG_fn", "HG_tpr", "HG_tnr",
                  "LG_AUC", "LG_cutoff", "LG_tp", "LG_tn", "LG_fp", "LG_fn", "LG_tpr", "LG_tnr"
                  ]

        df_result = pd.DataFrame(index=[events], columns=[columns])
        df_data.fillna(0, inplace=True)

        for event in events:
            print(event)
            print(file_name)

            each_results = []

            # 相関係数とp値の計算
            #print(df_data[event])
            print(df_result)
            # 欠損値を0で置き換える
            #スピアマンで計算しなおす
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

            # 平均と偏差の計算
            group_stats = df_data.groupby(set_group)[event].agg(['mean', 'std'])
            print(group_stats)
            df_result.loc[event, "HG_ave"] = group_stats.loc["HG", "mean"]
            #df_result.loc[event, "MG_ave"] = group_stats.loc["MG", "mean"]
            df_result.loc[event, "LG_ave"] = group_stats.loc["LG", "mean"]
            df_result.loc[event, "HG_sd"] = group_stats.loc["HG", "std"]
            #df_result.loc[event, "MG_sd"] = group_stats.loc["MG", "std"]
            df_result.loc[event, "LG_sd"] = group_stats.loc["LG", "std"]

            # Kruskal-Wallis test
            print(df_data[event].nunique())
            if not df_data[event].nunique()==1:
                kruskal_result = stats.kruskal(*[group[event] for _, group in df_data.groupby(set_group)])
                print("Kruskal-Wallis Test:")
                print("Test statistic:", kruskal_result.statistic)
                print("p-value:", kruskal_result.pvalue)

                df_result.loc[event, "group_p"] = kruskal_result.pvalue
            else:
                df_result.loc[event, "group_p"] = "NA"
                

            # Mann-Whitney U test with Bonferroni correction
            groups = df_data[set_group].unique()
            group_names = ["HG", "LG"]
            #group_names = list(groups)
            num_comparisons = len(groups) * (len(groups) - 1) // 2  # ペアごとの比較回数

            comparison_results = []

            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    group1 = df_data[df_data[set_group] == groups[i]][event]
                    group2 = df_data[df_data[set_group] == groups[j]][event]
                    mannwhitney_result = stats.mannwhitneyu(group1, group2)
                    corrected_pvalue = mannwhitney_result.pvalue * num_comparisons  # Bonferroni修正
                    group1_median = np.median(group1)
                    group2_median = np.median(group2)
                    #cohen_r = (group1_median - group2_median) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
                    r = mannwhitney_result.statistic / (len(group1) * len(group2))  # type1とtype2の間の効果量
                    cohen_r = r
                    comparison_results.append((group_names[i], group_names[j], corrected_pvalue, cohen_r))

                    if group_names[i] == "HG":
                        if group_names[j] == "MG":
                            df_result.loc[event, "p_HG-MG"] = corrected_pvalue
                            df_result.loc[event, "effect_r_HG-MG"] = cohen_r
                        elif group_names[j] == "LG":
                            df_result.loc[event, "p_HG-LG"] = corrected_pvalue
                            df_result.loc[event, "effect_r_HG-LG"] = cohen_r
                    elif group_names[i] == "MG":
                        if group_names[j] == "LG":
                            df_result.loc[event, "p_MG-LG"] = corrected_pvalue
                            df_result.loc[event, "effect_r_MG-LG"] = cohen_r

            #print(comparison_results)

            # Bonferroni修正後のp-valueを表示
            print("Corrected p-values:")
            for result in comparison_results:
                group1, group2, pvalue, cohen_r = result
                print(f"{group1} vs {group2}: p-value = {pvalue:.10f}, Cohen's r = {cohen_r:.6f}")
            
            '''
            group_names = ["HG", "LG"]
            for i in group_names:
                # データフレームから予測確率（またはスコア）と真のクラスラベルを取得
                print(df_data[set_group])
                print(i)
                y_true = (df_data[set_group]==i).astype(int)
                
                print(y_true)
                #df_data["{}_{}".format(i,set_group)] = (df_data[set_group]==i).astype(int)
                #print(df_data["{}_{}".format(i,set_group)] )
                #df_data["LG_{}".format(set_group)] = (df_data[set_group]=="LG").astype(int)

                #y_true = df_data['{}_{}'.format(i,set_group)]  # 真のクラスラベル
                y_score = df_data[event]  # モデルの予測確率またはスコア

                # ROC曲線を計算
                fpr, tpr, thresholds = roc_curve(y_true, y_score)

                # AUCを計算
                roc_auc = auc(fpr, tpr)

                # Youden Indexを使用して最適なカットオフ値を見つける
                youden_index = tpr - fpr
                best_cutoff_index = youden_index.argmax()
                best_cutoff = thresholds[best_cutoff_index]

                # カットオフ値を使用して混同行列を計算
                predicted_labels = (y_score >= best_cutoff).astype(int)
                confusion = confusion_matrix(y_true, predicted_labels)

                # 感度と特異度を計算
                tp = confusion[1, 1]  # True Positives
                tn = confusion[0, 0]  # True Negatives
                fp = confusion[0, 1]  # False Positives
                fn = confusion[1, 0]  # False Negatives

                sensitivity = tp / (tp + fn)  # 感度
                specificity = tn / (tn + fp)  # 特異度

                df_result.loc[event, "{}_AUC".format(i)] = roc_auc
                df_result.loc[event, "{}_cutoff".format(i)] = best_cutoff
                df_result.loc[event, "{}_tp".format(i)] = tp
                df_result.loc[event, "{}_tn".format(i)] = tn
                df_result.loc[event, "{}_fp".format(i)] = fp
                df_result.loc[event, "{}_fn".format(i)] = fn
                df_result.loc[event, "{}_tpr".format(i)] = sensitivity
                df_result.loc[event, "{}_tnr".format(i)] = specificity
            '''

            '''
            max_score = 0
            min_score = 0
            mid_scpre = 0

            if set_score == "score_f1":
                # データフレームから予測確率（またはスコア）と真のクラスラベルを取得
                max_score = 42
                min_score = 37
                mid_score = 40
            elif set_score == "score_f2":
                max_score = 35
                min_score = 30
                mid_score = 33
            elif set_score == "score_f3":
                max_score = 22
                min_score = 12
                mid_score = 18
            elif set_score == "score_all":
                max_score = 161
                min_score = 134
                mid_score = 150

            if max_score != 0:
                y_true = (df_data[set_score]>=max_score).astype(int)
                
                #print(y_true)
                y_score = df_data[event]  # モデルの予測確率またはスコア

                # ROC曲線を計算
                fpr, tpr, thresholds = roc_curve(y_true, y_score)

                # AUCを計算
                roc_auc = auc(fpr, tpr)

                # Youden Indexを使用して最適なカットオフ値を見つける
                youden_index = tpr - fpr
                best_cutoff_index = youden_index.argmax()
                best_cutoff = thresholds[best_cutoff_index]

                # カットオフ値を使用して混同行列を計算
                predicted_labels = (y_score >= best_cutoff).astype(int)
                confusion = confusion_matrix(y_true, predicted_labels)

                # 感度と特異度を計算
                tp = confusion[1, 1]  # True Positives
                tn = confusion[0, 0]  # True Negatives
                fp = confusion[0, 1]  # False Positives
                fn = confusion[1, 0]  # False Negatives

                sensitivity = tp / (tp + fn)  # 感度
                specificity = tn / (tn + fp)  # 特異度

                df_result.loc[event, "{}_AUC".format(max_score)] = roc_auc
                df_result.loc[event, "{}_cutoff".format(max_score)] = best_cutoff
                df_result.loc[event, "{}_tp".format(max_score)] = tp
                df_result.loc[event, "{}_tn".format(max_score)] = tn
                df_result.loc[event, "{}_fp".format(max_score)] = fp
                df_result.loc[event, "{}_fn".format(max_score)] = fn
                df_result.loc[event, "{}_tpr".format(max_score)] = sensitivity
                df_result.loc[event, "{}_tnr".format(max_score)] = specificity

                y_true = (df_data[set_score]<=min_score).astype(int)
                
                #print(y_true)
                #y_score = df_data[event]  # モデルの予測確率またはスコア

                # ROC曲線を計算
                fpr, tpr, thresholds = roc_curve(y_true, y_score)

                # AUCを計算
                roc_auc = auc(fpr, tpr)

                # Youden Indexを使用して最適なカットオフ値を見つける
                youden_index = tpr - fpr
                best_cutoff_index = youden_index.argmax()
                best_cutoff = thresholds[best_cutoff_index]

                # カットオフ値を使用して混同行列を計算
                predicted_labels = (y_score >= best_cutoff).astype(int)
                confusion = confusion_matrix(y_true, predicted_labels)

                # 感度と特異度を計算
                tp = confusion[1, 1]  # True Positives
                tn = confusion[0, 0]  # True Negatives
                fp = confusion[0, 1]  # False Positives
                fn = confusion[1, 0]  # False Negatives

                sensitivity = tp / (tp + fn)  # 感度
                specificity = tn / (tn + fp)  # 特異度

                df_result.loc[event, "{}_AUC".format(min_score)] = roc_auc
                df_result.loc[event, "{}_cutoff".format(min_score)] = best_cutoff
                df_result.loc[event, "{}_tp".format(min_score)] = tp
                df_result.loc[event, "{}_tn".format(min_score)] = tn
                df_result.loc[event, "{}_fp".format(min_score)] = fp
                df_result.loc[event, "{}_fn".format(min_score)] = fn
                df_result.loc[event, "{}_tpr".format(min_score)] = sensitivity
                df_result.loc[event, "{}_tnr".format(min_score)] = specificity

                y_true = (df_data[set_score]>=mid_score).astype(int)
                
                #print(y_true)
                #y_score = df_data[event]  # モデルの予測確率またはスコア

                # ROC曲線を計算
                fpr, tpr, thresholds = roc_curve(y_true, y_score)

                # AUCを計算
                roc_auc = auc(fpr, tpr)

                # Youden Indexを使用して最適なカットオフ値を見つける
                youden_index = tpr - fpr
                best_cutoff_index = youden_index.argmax()
                best_cutoff = thresholds[best_cutoff_index]

                # カットオフ値を使用して混同行列を計算
                predicted_labels = (y_score >= best_cutoff).astype(int)
                confusion = confusion_matrix(y_true, predicted_labels)

                # 感度と特異度を計算
                tp = confusion[1, 1]  # True Positives
                tn = confusion[0, 0]  # True Negatives
                fp = confusion[0, 1]  # False Positives
                fn = confusion[1, 0]  # False Negatives

                sensitivity = tp / (tp + fn)  # 感度
                specificity = tn / (tn + fp)  # 特異度

                df_result.loc[event, "{}_AUC".format(mid_score)] = roc_auc
                df_result.loc[event, "{}_cutoff".format(mid_score)] = best_cutoff
                df_result.loc[event, "{}_tp".format(mid_score)] = tp
                df_result.loc[event, "{}_tn".format(mid_score)] = tn
                df_result.loc[event, "{}_fp".format(mid_score)] = fp
                df_result.loc[event, "{}_fn".format(mid_score)] = fn
                df_result.loc[event, "{}_tpr".format(mid_score)] = sensitivity
                df_result.loc[event, "{}_tnr".format(mid_score)] = specificity
            '''
        df_result.to_excel("result_{}_rank_rate_{}.xlsx".format(file_name,set_group))
        #print(df_data)
