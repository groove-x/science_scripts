import pandas as pd
import os
import sys
import datetime



diary_csv_path = './00_base_data_diary/'  # 相対パスを指定
out_folder = "h_00_data_hugged_csv"

# フォルダ内の全ファイル名を取得
diary_csv_list = os.listdir(diary_csv_path)
diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]

#-----hugged

u_file = "HUGGED_bq-results-20240815-034518-1723693533880"

df_data_u = pd.read_csv("{}/{}.csv".format(diary_csv_path, u_file), sep=',')
df_data_u['Timestamp'] = pd.to_datetime(df_data_u['Timestamp'])
df_data_u['Timestamp_end'] = pd.to_datetime(df_data_u['Timestamp_end'])
u_ghost = df_data_u["GhostID"].unique()
line_number = len(u_ghost)

for csv_file in diary_csv_files:
    file_path = os.path.join(diary_csv_path, csv_file)
    df_data = pd.read_csv(file_path)

    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])

    df_ex_data = pd.DataFrame()

    all_ex_data = []


    l = 0
    for l_ghost in u_ghost:
    #for row in df_charger.itertuples(index=False):
        l = l+1    
        print("ghost_num:{}/{}".format(l, line_number))
        df_w = df_data[df_data['GhostID']==l_ghost]
        df_t = df_data_u[df_data_u['GhostID']==l_ghost]
        print(l_ghost)
        #print(df)
        line_n = len(df_t)
        ln = 0

         # ベクトル化された操作に変更
        timestamps = df_t['Timestamp']
        timestamps_end = df_t['Timestamp_end']

        for ln, (timestamp, timestamp_end) in enumerate(zip(timestamps, timestamps_end), start=1):

            #for row in df_t.itertuples(index=False):
            #ln = ln+1
            print("ghost_num:{}/{}".format(l, line_number))
            print("line_num:{}/{}".format(ln, line_n))

            one_hour_ago = timestamp - pd.Timedelta(hours=1)


            # 条件フィルタリング
            df_one = df_w[(df_w['Timestamp'] <= timestamp_end) & (df_w['Timestamp'] >= timestamp)]
            all_ex_data.append(df_one)

            '''
            #one_hour_ago = row.Timestamp - pd.Timedelta(hours=1)
            df_one = df_w[(df_w['Timestamp']<=row.Timestamp_end) & (df_w['Timestamp']>=row.Timestamp)]
            df_ex_data = pd.concat([df_ex_data, df_one], axis=0,ignore_index=True)


            #event_ago = row.Timestamp - pd.Timedelta(minutes=1)
            df_one = df_w[(df_w['Timestamp']>=event_ago) & (df_w['Timestamp']<row.Timestamp)]
            df_ex_after_data = pd.concat([df_ex_after_data, df_one], axis=0,ignore_index=True)

            #event_later = row.Timestamp_end + pd.Timedelta(minutes=1)
            df_one = df_w[(df_w['Timestamp']<=event_later) & (df_w['Timestamp']>row.Timestamp_end)]
            df_ex_before_data = pd.concat([df_ex_before_data, df_one], axis=0,ignore_index=True)
            '''
    # 一度だけconcatを使用してデータフレームを結合
    print("output")
    if all_ex_data:
        df_ex_data = pd.concat(all_ex_data, axis=0, ignore_index=True)
    print(df_ex_data)
    df_ex_data.to_csv("./{}/hugged_{}".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

