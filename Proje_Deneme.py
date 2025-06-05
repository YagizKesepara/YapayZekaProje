import tkinter as tk
import pandas as pd
from tkinter import filedialog

# file_path = filedialog.askopenfilename()
# df = pd.read_csv(file_path)

import seaborn as sns 
import matplotlib.pyplot as plt

def dosya_sec():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        print("Ilk 5 Satir")
        print(df.head)

root=tk.Tk()
root.title("Veri Yukleme Arayuzu")
root.geometry("400x200")
root.configure(bg='white')

buton = tk.Button(root, text="Veri Dosyası Seç", command=dosya_sec, bg='lightblue', fg='black', font=("Arial", 12))
buton.pack(pady=60)

root.mainloop()