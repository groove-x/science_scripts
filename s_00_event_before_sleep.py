import pandas as pd
import os
import sys
import datetime


diary_csv_path = './00_base_data_diary/'  # 相対パスを指定
out_folder = "s_00_data_sleep_wake_csv"

# フォルダ内の全ファイル名を取得
diary_csv_list = os.listdir(diary_csv_path)
diary_csv_files = [file for file in diary_csv_list if file.endswith('.csv')]

#-----sleep

u_file = "02_SLEEP_bq-results-20240808-055018-1723096245038"

df_data_u = pd.read_csv("{}/sleep_wakeup/{}.csv".format(diary_csv_path, u_file), sep=',')
df_data_u['Timestamp'] = pd.to_datetime(df_data_u['Timestamp'])
u_ghost = df_data_u["GhostID"].unique()
line_number = len(u_ghost)

for csv_file in diary_csv_files:
    file_path = os.path.join(diary_csv_path, csv_file)
    df_data = pd.read_csv(file_path)

    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])

    df_ex_data = pd.DataFrame()

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
        for row in df_t.itertuples(index=False):
            ln = ln+1
            print("ghost_num:{}/{}".format(l, line_number))
            print("line_num:{}/{}".format(ln, line_n))

            one_hour_ago = row.Timestamp - pd.Timedelta(hours=1)
            df_one = df_w[(df_w['Timestamp']>=one_hour_ago) & (df_w['Timestamp']<=row.Timestamp)]
            df_ex_data = pd.concat([df_ex_data, df_one], axis=0,ignore_index=True)


    df_ex_data.to_csv("./{}/sleep_1hr_{}".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない


#-----wake

u_file = "01_WAKEUP_bq-results-20240808-055202-1723096332305"

df_data_u = pd.read_csv("{}/sleep_wakeup/{}.csv".format(diary_csv_path, u_file), sep=',')
df_data_u['Timestamp'] = pd.to_datetime(df_data_u['Timestamp'])
u_ghost = df_data_u["GhostID"].unique()
line_number = len(u_ghost)

for csv_file in diary_csv_files:
    file_path = os.path.join(diary_csv_path, csv_file)
    df_data = pd.read_csv(file_path)

    df_data['Timestamp'] = pd.to_datetime(df_data['Timestamp'])

    df_ex_data = pd.DataFrame()

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
        for row in df_t.itertuples(index=False):
            ln = ln+1
            print("ghost_num:{}/{}".format(l, line_number))
            print("line_num:{}/{}".format(ln, line_n))
            one_hour_later = row.Timestamp + pd.Timedelta(hours=1)
            df_one = df_w[(df_w['Timestamp']<=one_hour_later) & (df_w['Timestamp']>=row.Timestamp)]
            df_ex_data = pd.concat([df_ex_data, df_one], axis=0,ignore_index=True)


    df_ex_data.to_csv("./{}/wake_1hr_{}".format(out_folder, csv_file), index=False)  # index=False により、インデックスはCSVに含まれない

