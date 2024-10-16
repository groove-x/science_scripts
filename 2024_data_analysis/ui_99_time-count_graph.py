import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, StringVar, filedialog, OptionMenu
from tkinter import messagebox

def load_file():
    """ファイルを選択し、データを読み込む"""
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")])
    if not file_path:
        return

    global df
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)

        print(df)
        
        # UIのIDメニューを更新
        update_id_menu()
        status_var.set(f"ファイル読み込み成功: {file_path}")
    except Exception as e:
        messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました。\n{e}")

def update_id_menu():
    """IDメニューを更新"""
    id_list = df['GhostID'].unique()
    id_var.set(id_list[0])  # デフォルト選択
    id_menu['menu'].delete(0, 'end')
    for id in id_list:
        id_menu['menu'].add_command(label=id, command=lambda value=id: id_var.set(value))

def plot_graph():
    """選択したIDのグラフを表示"""
    selected_id = id_var.get()
    filtered_df = df[df['GhostID'] == selected_id]
    u_event = df['Event'].unique()

    if filtered_df.empty:
        messagebox.showwarning("警告", "選択したIDのデータがありません。")
        return

    plt.figure()
    plt.plot(filtered_df['every_days_unit'], filtered_df['every_days_sum'], marker='o', linestyle='-')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.title('ID: {}_{}'.format(selected_id, u_event[0]))
    #plt.title(f'ID: {selected_id}_{u_event[0]}')
    plt.grid(True)
    plt.show()

# GUIの設定
root = Tk()
root.title("データ可視化ツール")
root.geometry("400x400")

status_var = StringVar()
status_var.set("ファイルを読み込んでください。")

load_button = Button(root, text="ファイルを読み込む", command=load_file)
load_button.pack(pady=5)

id_var = StringVar()
id_menu = OptionMenu(root, id_var, [])
id_menu.pack(pady=5)

plot_button = Button(root, text="グラフを作成", command=plot_graph)
plot_button.pack(pady=5)

status_label = Label(root, textvariable=status_var)
status_label.pack(pady=5)

root.mainloop()
