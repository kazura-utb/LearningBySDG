import sys
import os
import pickle
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt 
import joblib
from sklearn.linear_model import SGDRegressor
from scipy.sparse import csr_matrix
from sklearn import preprocessing

args = sys.argv
if len(args) < 2:
    print("Please input stage.");
    sys.exit()

stage = args[1]
if int(stage, 10) < 0 or int(stage, 10) > 60:
    print("Range of stage is [0 - 60].");
    sys.exit()

filename_data = 'train_data/train_data_' + stage + '.zip'
if not os.path.exists(filename_data) :
    print("Training file is Not Found.");
    sys.exit()

filename_col = 'train_data/train_col_' + stage + '.zip'
if not os.path.exists(filename_col) :
    print("Training file is Not Found.");
    sys.exit()

filename_row = 'train_data/train_row_' + stage + '.zip'
if not os.path.exists(filename_row) :
    print("Training file is Not Found.");
    sys.exit()

filename_obj = 'train_data/train_obj_' + stage + '.zip'
if not os.path.exists(filename_obj) :
    print("Training file is Not Found.");
    sys.exit()

# Read CSV
data = dd.read_csv(filename_data, compression='zip', blocksize=None, header=None, sep=',', dtype=int).compute().values.ravel()
print(data)

col  = dd.read_csv(filename_col, compression='zip', blocksize=None, header=None, sep=',', dtype=int).compute().values.ravel()
print(col)

row  = dd.read_csv(filename_row, compression='zip', blocksize=None, header=None, sep=',', dtype=int).compute().values.ravel()
print(row)

obj  = dd.read_csv(filename_obj, compression='zip', blocksize=None, header=None, sep=',', dtype=int).compute().values.ravel()
print(obj)

# 疎行列に変換
X = csr_matrix((data, (row, col)), dtype=int)
# SGDアルゴリズムで学習
sgd_reg = SGDRegressor(max_iter = 10000, fit_intercept = False).fit(X, obj)

a = sgd_reg.coef_
print("a:", a)

# ndarrayをリストに変換して文字列で書き出し
list_ = a.tolist()
l_n_str = [str(int(n * 1024)) for n in list_]

str_ = '\n'.join(l_n_str)
with open('result/' + stage + '.txt', 'wt') as f:
    f.write(str_)


Y_pred=sgd_reg.predict(X)
#plt.scatter(obj, obj - Y_pred, c='r', marker='s', s=4, label="ALL")
#plt.legend()
#plt.hlines(y=0, xmin=-64, xmax=64, colors='black')
#plt.show()

# 二乗誤差表示
RMS=np.mean((Y_pred - obj) ** 2)
print(RMS)

