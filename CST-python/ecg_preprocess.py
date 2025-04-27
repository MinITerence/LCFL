import os
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt

root_path = './data/ARDB'

# download MIT-BIT dataset
if not os.path.exists(os.path.join(root_path)):
    wfdb.dl_database('mitdb', os.path.join(root_path))
else:
    print('You already have the MIT-BIH data.')


rm_patient = [102, 104, 107, 114, 217]  # patients to remove
files = os.listdir(os.path.join(root_path))
patients = [int(f[0:-4]) for f in files if f[-4:] != 'hdf5']
patients = list(set(patients))
patients.sort()  # list of all patients
for p in rm_patient:
    patients.remove(p)
print(patients)

def is_valid_sample(left, cur, right, interval, siglen):
    left_val = cur - interval >= 0 and cur - interval >= left
    right_val = cur + interval <= siglen - 1 and cur + interval <= right
    return left_val and right_val

def denoise(signal):
    thr = 0.005 * np.sqrt(2 * np.log2(len(signal)))
    c = pywt.wavedec(signal, 'bior4.4')
    thr_func = lambda x : pywt.threshold(x, thr, 'soft')
    nc = list(map(thr_func, c))
    return pywt.waverec(nc, 'bior4.4')

interval = 100  # length to slice from R-peak

# make [N, L, R, A, V] classified data
x = []
y = []
label_list = ['N', 'L', 'R', 'A', 'V']
for p in patients:
    samp = wfdb.rdsamp(os.path.join(root_path, str(p)))
    signal = np.array(samp[0][:, 0])  # second lead beats of a patient
    signal = (signal - min(signal)) / (max(signal) - min(signal))  # normalize
    ann = wfdb.rdann(os.path.join(root_path, str(p)), 'atr')  # annotations of each beat
    r_peaks = ann.sample[1:-1]  # get R-peaks of each beat
    labels = ann.symbol[1:-1]  # get labels of each beat
    for i in range(1, len(r_peaks) - 1):
        if is_valid_sample(r_peaks[i-1], r_peaks[i], r_peaks[i+1], interval, len(signal)) and labels[i] in label_list:
            beat = signal[r_peaks[i] - interval:r_peaks[i] + interval + 1]
            assert len(beat) == 2 * interval + 1
            x.append(beat)
            y.append(label_list.index(labels[i]))

from scipy.signal import resample

# downsample the signal
sample_size = 128
for i in range(len(x)):
    x[i] = (x[i] - min(x[i])) / (max(x[i]) - min(x[i]))  # normalize
    x[i] = resample(x[i], sample_size)
    x[i] = denoise(x[i])  # denoise

from collections import Counter

x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)
print(Counter(y))


# make it balanced
# sample_list = [[], [], [], [], []]
# for i in range(len(x)):
#     sample_list[y[i]].append(x[i])
# sample_list = np.array(sample_list)

# np.random.seed(2019)
# num_to_sample = [6000, 6000, 6000, 2490, 6000]  # from [N, L, R, A, V]
# for i in range(len(num_to_sample)):
#     idx = np.random.choice(len(sample_list[i]), num_to_sample[i])
#     sample_list[i] = np.array(sample_list[i])[idx]
#     print(sample_list[i].shape)

# x_train = None
# y_train = None
# x_test = None
# y_test = None

# # make train and test set
# for i in range(len(num_to_sample)):
#     if x_train is not None:
#         x_train = np.concatenate((x_train, sample_list[i][:int(num_to_sample[i] / 2)]))
#         y_train += [i] * int(num_to_sample[i] / 2)
#         x_test = np.concatenate((x_test, sample_list[i][int(num_to_sample[i] / 2):]))
#         y_test += [i] * int(num_to_sample[i] / 2)
#     else:
#         x_train = sample_list[i][:int(num_to_sample[i] / 2)]
#         y_train = [i] * int(num_to_sample[i] / 2)
#         x_test = sample_list[i][int(num_to_sample[i] / 2):]
#         y_test = [i] * int(num_to_sample[i] / 2)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# # shuffle train and test set
# idx = np.arange(len(x_train))
# np.random.shuffle(idx)
# x_train = x_train[idx]
# y_train = y_train[idx]
# np.random.shuffle(idx)
# x_test = x_test[idx]
# y_test = y_test[idx]

# print(x_train.shape)
# print(y_train.shape)
# print(Counter(y_train))
# print(x_test.shape)
# print(y_test.shape)
# print(Counter(y_test))

# # create num_channels dimension
# x = np.expand_dims(x, axis=1)
# x_train = np.expand_dims(x_train, axis=1)
# x_test = np.expand_dims(x_test, axis=1)








##使用scikit-learn库中train_test_split函数来划分数据
from sklearn.model_selection import train_test_split

# 使用train_test_split划分数据集
train_size = 0.6  # 训练集占总数据集的比例
test_size = 0.2   # 测试集占总数据集的比例
validation_size = 0.2  # 验证集占总数据集的比例
 
# 合并测试集和验证集的比例，因为train_test_split不直接支持划分三个集
validation_size += test_size
 
# 划分数据集
train_X, test_and_validation_X, train_y, test_and_validation_y = train_test_split(
    x, y, test_size=validation_size, random_state=42)
 
# 进一步划分测试集和验证集
validation_X, test_X, validation_y, test_y = train_test_split(
    test_and_validation_X, test_and_validation_y, test_size=0.5, random_state=42)
 
# 打印集的大小
print("训练集大小:", len(train_X))
print("验证集大小:", len(validation_X))
print("测试集大小:", len(test_X))



print(train_X.shape)
print(train_y.shape)
print(Counter(train_y))

print(validation_X.shape)
print(validation_y.shape)
print(Counter(validation_y))

print(test_X.shape)
print(test_y.shape)
print(Counter(test_y))

# create num_channels dimension
x = np.expand_dims(x, axis=1)
train_X = np.expand_dims(train_X, axis=1)
validation_X = np.expand_dims(validation_X, axis=1)
test_X = np.expand_dims(test_X, axis=1)
print(x.shape)
print(train_X.shape)
print(validation_X.shape)
print(test_X.shape)




import h5py

# save to hdf5 file
with h5py.File('./mitdb/all.hdf5', 'w') as hdf:
    hdf['x'] = x[:]
    hdf['y'] = y[:]
    print('All data saved to all.hdf5')
with h5py.File('./mitdb/train.hdf5', 'w') as hdf:
    hdf['x_train'] = train_X[:]
    hdf['y_train'] = train_y[:]
    print('Train data saved to train.hdf5')
with h5py.File('./mitdb/val.hdf5', 'w') as hdf:
    hdf['x_val'] = validation_X[:]
    hdf['y_val'] = validation_y[:]
    print('Validation data saved to val.hdf5')    
with h5py.File('./mitdb/test.hdf5', 'w') as hdf:
    hdf['x_test'] = test_X[:]
    hdf['y_test'] = test_y[:]
    print('Test data saved to test.hdf5')

# import h5py

# # save to hdf5 file
# with h5py.File(os.path.join(root_path, 'all_ecg.hdf5'), 'w') as hdf:
#     hdf['x'] = x[:]
#     hdf['y'] = y[:]
#     print('All data saved to all_ecg.hdf5')
# with h5py.File(os.path.join(root_path, 'train_ecg.hdf5'), 'w') as hdf:
#     hdf['x_train'] = x_train[:]
#     hdf['y_train'] = y_train[:]
#     print('Train data saved to train_ecg.hdf5')
# with h5py.File(os.path.join(root_path, 'test_ecg.hdf5'), 'w') as hdf:
#     hdf['x_test'] = x_test[:]
#     hdf['y_test'] = y_test[:]
#     print('Test data saved to test_ecg.hdf5')
