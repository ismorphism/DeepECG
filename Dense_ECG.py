from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from biosppy.signals import ecg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pandas as pd
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
import keras
from keras import regularizers
from matplotlib import pyplot as plt

np.random.seed(7)

number_of_classes = 4


def change(x):  # Для получения чисел от 0 до 3
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


mypath = 'training2017/'
onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f[0] == 'A')]
bats = [f for f in onlyfiles if f[7] == 'm']
check = 3000
mats = [f for f in bats if (np.shape(sio.loadmat(mypath + f)['val'])[1] >= check)]
size = len(mats)
print('Training size is ', len(mats))
X = np.zeros((len(mats), check))
for i in range(len(mats)):
    X[i, :] = sio.loadmat(mypath + mats[i])['val'][0, :check]

target_train = np.zeros((len(mats), 1))
Train_data = pd.read_csv(mypath + 'REFERENCE.csv', sep=',', header=None, names=None)
for i in range(len(mats)):
    if Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'N':
        target_train[i] = 0
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'A':
        target_train[i] = 1
    elif Train_data.loc[Train_data[0] == mats[i][:6], 1].values == 'O':
        target_train[i] = 2
    else:
        target_train[i] = 3

Label_set = np.zeros((len(mats), number_of_classes))
for i in range(np.shape(target_train)[0]):
    dummy = np.zeros((number_of_classes))
    dummy[int(target_train[i])] = 1
    Label_set[i, :] = dummy

inputs = 60  # Previus value for 9k check is 95
X_new = np.zeros((size, inputs))
for i in range(size):
    out = ecg.christov_segmenter(signal=X[i, :], sampling_rate=300.)
    A = np.hstack((0, out[0][:len(out[0]) - 1]))
    B = out[0]
    dummy = np.lib.pad(B - A, (0, inputs - len(B)), 'constant', constant_values=(0))
    X_new[i, :] = dummy

print('All is OK')
X = X_new
X = (X - X.mean()) / (X.std())
Label_set = Label_set[:size, :]


# X_new = np.zeros((size, check))
# Label_new = np.zeros((size, 4))
# stop = 0
# j = -1
# for i in range(np.shape(X)[0]):
#     if (stop == 1000) and (np.array_equal(Label_set[i, :], [1, 0, 0, 0])):
#         continue
#     else:
#         j += 1
#         if j != size:
#             if np.array_equal(Label_set[i, :], [1, 0, 0, 0]):
#                 stop += 1
#             X_new[j, :] = X[i, :]
#             Label_new[j, :] = Label_set[i, :]
#         else:
#             break
#
# X = X_new
# Label_set = Label_new[:, :]

# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)


def train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i):
    checkpointer = ModelCheckpoint(filepath='Dense_models/Best_model of ' + str(i) + '.h5', monitor='val_acc',
                                   verbose=1, save_best_only=True)
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
    hist = model.fit(X_train, Y_train, epochs=500, batch_size=256, validation_data=(X_val, Y_val), verbose=2,
                     shuffle=True, callbacks=[checkpointer])
    pd.DataFrame(hist.history).to_csv(path_or_buf='Dense_models/History ' + str(i) + '.csv')
    model.save('my_model ' + str(i) + '.h5')
    predictions = model.predict(X_val)
    score = accuracy_score(change(Y_val), change(predictions))
    print(score)
    df = pd.DataFrame(change(predictions))
    df.to_csv(path_or_buf='Dense_models/Preds_' + str(format(score, '.4f')) + '__' + str(i) + '.csv', index=None,
              header=None)
    model.save('Dense_models/' + str(format(score, '.4f')) + '__' + str(i) + '_my_model.h5')
    pd.DataFrame(confusion_matrix(change(Y_val), change(predictions))).to_csv(
        path_or_buf='Dense_models/Result_Conf' + str(format(score, '.4f')) + '__' + str(i) + '.csv', index=None,
        header=None)


def create_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(inputs,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


skf = StratifiedKFold(n_splits=10, shuffle=True)
target_train = target_train.reshape(size, )
# print(skf.get_n_splits(X, target_train))
# print(skf.split(X, target_train))
for i, (train_index, test_index) in enumerate(skf.split(X, target_train)):
    print("TRAIN:", train_index, "TEST:", test_index)
    # train = 0.9
    # print('Training_size is ', int(train*size))
    # print('Validation_size is ', size - int(train*size))
    X_train = X[train_index, :]
    Y_train = Label_set[train_index, :]
    X_val = X[test_index, :]
    Y_val = Label_set[test_index, :]
    # X_train = X[:int(train*size), :]
    # Y_train = Label_set[:int(train*size), :]
    # X_val = X[int(train*size):, :]
    # Y_val = Label_set[int(train*size):, :]
    # model = None
    model = create_model()
    train_and_evaluate__model(model, X_train, Y_train, X_val, Y_val, i)
