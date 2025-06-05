import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tkinter import messagebox

class Uygulama(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crop Recommendation - Data Analysis App")
        self.geometry("500x300")
        self.df = None
        self.y_axis_var = tk.StringVar()
        self.x_axis_var = tk.StringVar()
        self.model = None
        self.le = None
        self.show_ana_ekran()
    
    def show_ana_ekran(self):
        self.clear_window()

        label = tk.Label(self, text="Select CSV Dataset", font=("Arial", 14))
        label.pack(pady=20)

        btn = tk.Button(self, text="Choose File", command=self.dosya_sec, font=("Arial", 12))
        btn.pack()

    def dosya_sec(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            print(self.df.head())
            self.show_analiz_ekran()

    def show_analiz_ekran(self):
        self.clear_window()

        label = tk.Label(self, text="EDA Panel", font=("Arial", 14))
        label.pack(pady=10)

        numeric_cols = list(self.df.select_dtypes(include='number').columns)
        if 'label' in numeric_cols:
            numeric_cols.remove('label')

        #X Ekseni
        x_label = tk.Label(self, text="Select X Axis")
        x_label.pack()
        x_combo = ttk.Combobox(self, textvariable=self.x_axis_var, values=numeric_cols)
        x_combo.pack()

        #Y Eksni
        y_label = tk.Label(self, text="Select Y Axis")
        y_label.pack()
        y_combo = ttk.Combobox(self, textvariable=self.y_axis_var, values=numeric_cols)
        y_combo.pack()

        btn_scatter = tk.Button(self, text="Scatter Plot", command=self.plot_scatter)
        btn_scatter.pack(pady=5)

        btn_box = tk.Button(self, text="Box Plot", command=self.plot_box)
        btn_box.pack(pady=5)

        btn_corr = tk.Button(self, text="Correlation Heatmap", command=self.plot_corr)
        btn_corr.pack(pady=5)

        btn_target = tk.Button(self, text="Target Variable Distirbution", command=self.plot_target)
        btn_target.pack(pady=5)

        btn_train = tk.Button(self, text="Train Model", command=self.train_model)
        btn_train.pack(pady=5)

    def plot_scatter(self):
        if self.df is not None:
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()

        if x_col and y_col:
            self.df.plot.scatter(x=x_col, y=y_col)
            plt.title(f"Scatter Plot : {x_col} vs {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("Please Select Both X and Y Axis")

    def plot_box(self):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include='number').columns
            clean_cols = [col for col in numeric_cols if col != "label" and self.df[col].max() < 1e6]

            if clean_cols:
                self.df[clean_cols].plot(kind="box", figsize=(8, 5))
                plt.title("Box Plot")
                plt.tight_layout()
                plt.show()
            else:
                print("No valid columns for box plot.")

    def plot_corr(self):
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include='number').drop(columns=['label'], errors='ignore')
            corr = numeric_cols.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

    def plot_target(self):
        if self.df is not None and "label" in self.df.columns:
            counts = self.df["label"].value_counts()
            counts.plot(kind = 'bar', color= 'skyblue')
            plt.title("Distirbution of Target Variable (label)")
            plt.xlabel("Crop Label")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("Target Variable 'label' not found in dataset.")

    def train_model(self):
        if self.df is not None:
            try:
                x = self.df.drop(columns=["label"])
                y = self.df["label"]

                #Label Encoding
                self.le = LabelEncoder()
                y_encoded = self.le.fit_transform(y)

                #Data Split
                X_train, X_test, y_train, y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42, stratify=y)

                #Model Training
                self.model = RandomForestClassifier(n_estimators=100,random_state=42)
                self.model.fit(X_train,y_train)

                #Prediction
                y_pred = self.model.predict(X_test)
                acc = accuracy_score(y_test,y_pred)
                print(f"Model Trained Succesfully Accuracy: {acc:.2f}")

                tk.messagebox.showinfo("Training Complete", f"Model trained successfully!\nAccuracy: {acc:.2f}")
            except Exception as e:
                print("Error During Training",e)
                tk.messagebox.showerror("Error", str(e))
    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

# Start app
app = Uygulama()
app.mainloop()
