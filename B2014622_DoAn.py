import pandas as pd
import numpy as np
import seaborn as sns;
import matplotlib.pylab as plt
dt = pd.read_excel('Folds5x2_pp.xlsx')
print("Kiem tra null")
print(dt.isnull().sum())
print("--------------------\n")

from sklearn.preprocessing import MinMaxScaler

X= dt.iloc[:,0:4]
y= dt.PE
scaler = MinMaxScaler()
dulieu = scaler.fit_transform(X)
dulieu = np.array(dulieu)
print(dulieu)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)
predict_knn = knn.predict(X_test)
print('MAE: ', metrics.mean_absolute_error(y_test, predict_knn))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_knn)))
print("--------------------\n")

print(" Linear regression")


from sklearn.linear_model import LinearRegression
from sklearn import metrics
reg = LinearRegression()
reg.fit(X_train,y_train)
predict_reg = reg.predict(X_test)
print('MAE: ', metrics.mean_absolute_error(y_test, predict_reg))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_reg)))
print("--------------------\n")

# plt.scatter(X_train,y_train,'o',color='blue')
# plt.show()

print("Decision Tree")
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
clf = DecisionTreeRegressor(max_depth=12 , random_state=42, min_samples_leaf=5 )
clf.fit(X_train,y_train)
predict_tree = clf.predict(X_test)
print('MAE: ' ,metrics.mean_absolute_error(y_test, predict_tree))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict_tree)))
print("--------------------\n")

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf)
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset Wind Speed Prediction vào biến data

# Chạy vòng lặp for để lặp lại quá trình huấn luyện và kiểm tra mô hình
accuracies = []
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
A_train, A_test, b_train, b_test = train_test_split(X, y, test_size=0.3, random_state=42)
MAE_KNN=0
MAE_DES=0
MAE_REG=0
for i in range(10):
    # Khởi tạo mô hình Decision Tree với các tham số Smặc định
    model = DecisionTreeRegressor(max_depth=12 + i , random_state=42, min_samples_leaf=5 + i)
    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(A_train, b_train)
    # Dự đoán nhãn trên tập kiểm tra
    predict_des = model.predict(A_test)
    # Tính độ chính xác của mô hình
    print("--------------------\n")
    print("Do chinh xac:", i + 1)
    print("Max_depth:", 12 + i)
    print("Min_samples_leaf:", 5 + i)
    print("MAE DecisionTree: ", metrics.mean_absolute_error(b_test, predict_des))
    knn = KNeighborsRegressor(n_neighbors=9 + i)
    knn.fit(A_train, b_train)
    predict_knn = knn.predict(A_test)
    print('MAE KNN: ', metrics.mean_absolute_error(b_test, predict_knn))
    reg = LinearRegression()
    reg.fit(A_train, b_train)
    predict_reg = reg.predict(A_test)
    print('MAE Linear regression: ', metrics.mean_absolute_error(b_test, predict_reg))
    MAE_KNN+=metrics.mean_absolute_error(b_test, predict_knn)
    MAE_DES+=metrics.mean_absolute_error(b_test, predict_des)
    MAE_REG+=metrics.mean_absolute_error(b_test, predict_reg)

print('Do lech chuan trung binh cua thuat toan KNN',MAE_KNN/10)
print('Do lech chuan trung binh cua thuat toan Decision Tree',MAE_DES/10)
print('Do lech chuan trung binh cua thuat toan Linear Regression',MAE_REG/10)

label = ["KNN","Decision Tree","Linear Regression"]
value = [MAE_KNN/10,MAE_DES/10,MAE_REG/10]

plt.figure(figsize = (5, 3))
plt.bar(label,value,width=0.2, color="firebrick")
plt.xlabel("Thuật toán")
plt.ylabel("Độ lệch chuẩn trung bình")
plt.title("Độ lệch chuẩn trung bình của các thuật toán")
plt.show()