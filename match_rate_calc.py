import csv
import numpy as np

test_file_dir = "C:\so_compli\src\submission.csv"
sol_file_dir = "C:\so_compli\src\cnn_mnist_datagen.csv"

test_file = np.loadtxt(test_file_dir, skiprows=1, dtype='int', delimiter=',')
sol_file = np.loadtxt(sol_file_dir, skiprows=1, dtype='int', delimiter=',')

wrong = 0

for idx in range(28000):
    if test_file[idx][1] != sol_file[idx][1]:
        wrong += 1
        print(test_file[idx], sol_file[idx])
match_rate = (28000-wrong)/28000*100
print("wrong : "+str(wrong)+'\n'+"match rate : "+str(match_rate)+'%')