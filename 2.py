import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import math
# special matplotlib argument for improved plots
from matplotlib import rcParams
tenCot = ['SO_DON', 'TIEN_THUONG', 'TIEN_THUCTE ']
df = pd.read_csv("D:/hetmon/hetmon/data.csv", names= tenCot)
maTran= df.values 
m,n= maTran.shape   # m:=47(số hàng); n:=3 (số cột)
X = maTran[:,0:n-1]
X = np.insert(X,0, values =1, axis = 1) 
y = maTran[:,n-1:n]
print(df)
print('Xtrain = ',X)
print('ytrain = ',y)
w = np.linalg.lstsq(X,y)[0]

w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
print('w = ',w)
df1 = pd.read_csv('2.csv', names= tenCot)

X_test= df1.values
m,n= X_test.shape 
X_test= np.c_[X_test, np.ones(X_test.shape[0])] 
XX = X_test[:,0:n-1]
XX = np.insert(XX,0, values =1, axis = 1)
y_true = X_test[:,n-1:n]                         
Y_pred=np.dot(XX,w)
print('Xtest = ',XX)
print('Y_test=',y_true)
print('Y_pred=',Y_pred)

u=abs(y_true - Y_pred)
A=(u.sum())/(m-1)
A=round(A,2)
# print('u',u)
print('Sai số dự đoán',A)
master=tk.Tk()
master.title("Dự đoán lương")
tk.Label(master, text="Nhập các thông tin dưới đây").grid (column=0, row=0)
tk.Label (master, text="Số đơn").grid (column=0, row=1)
s1 = Entry(master,width=50)
s1.grid(column=1, row=1)
tk.Label (master, text="Tiền thưởng").grid (column=0, row=2)
s2 = Entry(master,width=50)
s2.grid(column=1, row=2)
tk.Button(master, 
          text='Thoát', 
          command=master.quit).grid(row=4, 
                                    column=0, 
                                    sticky=tk.W,
                                    pady=4)
def predict():
	bien1=float(s1.get())
	bien2=float(s2.get())
	y_0 = w_0 + w_1*bien1+w_2*bien2
	y_0=round(y_0,3)
	messagebox.showinfo( "Tiền Lương dự đoán:",y_0)   
tk.Button(master,  text='Hiển thị giá dự đoán', command=predict).grid(row=4, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=4)
master.mainloop()