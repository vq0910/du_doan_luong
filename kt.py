import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce
import sys
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

df = pd.read_csv("D:/hetmon/hetmon/data.csv")
data_X = df.iloc[:, 1:-1]
data_y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(data_X, data_y, test_size=0.3, shuffle=False)
encoder = ce.OrdinalEncoder(cols=['SO_DON', 'TIEN_THUONG'])
xtrain = encoder.fit_transform(xtrain)
xtest = encoder.fit_transform(xtest)

svc = SVC()  # kernel = 'linear'

svc.fit(xtrain, ytrain)

data_x = svc.predict(xtest)
data_y = np.array(ytest)
count = 0
for i in range(len(data_x)):
    if (data_x[i] == data_y[i]):
        count = count + 1
    print(i, 'Kết quả dự đoán :', data_x[i], ', Thực tế :', data_y[i])
rate = round((count / len(data_x)) * 100)
print('Support-Vector-Machine cho ta tỉ lệ dự đoán như sau : ')
print('Số dự đoán đúng', count, 'trên tổng', len(data_x), '\nTỷ lệ đúng đạt :', rate, '%')
# y_svc = svc.predict(X_test)
# print('svc:', X_test)


form = Tk()
form.title("Dự đoán tiền lương:")
form.geometry("450x350")

lable_ten = Label(form, text="Nhập thông tin:")
lable_ten.grid(row=1, column=2, padx=40, pady=10)

lable_CodingHours = Label(form, text=" Số đơn:")
lable_CodingHours.grid(row=2, column=2, padx=40, pady=10)
textbox_CodingHours = Entry(form)
textbox_CodingHours.grid(row=2, column=3)

lable_CodingHours = Label(form, text=" Tiền thưởng:")
lable_CodingHours.grid(row=3, column=2, padx=40, pady=10)
textbox_CodingHours_1 = Entry(form)
textbox_CodingHours_1.grid(row=3, column=3)


def dudoan():
    CodingHours = textbox_CodingHours.get()
    CoffeeTime = textbox_CodingHours_1.get()

    X_dudoan = np.array([CodingHours, CoffeeTime]).reshape(1, -1)

    y_kqua = svc.predict(X_dudoan)
    messagebox.showinfo("Kết quả dự đoán", str(y_kqua))


button_svm = Button(form, text='Kết quả dự đoán heo SVM', command=dudoan)
button_svm.grid(row=7, column=2, pady=20)


def khanang():
    y_svc = svc.predict(xtest)
    messagebox.showinfo("Khả năng dự đoán của SVM",
                        "Độ chính xác của phương pháp: " + str(accuracy_score(ytest, y_svc) * 100) + "%")


button_svm1 = Button(form, text='Khả năng dự đoán ', command=khanang)
button_svm1.grid(row=7, column=3, padx=30)

form.mainloop()