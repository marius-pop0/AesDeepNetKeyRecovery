import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


X_train = np.loadtxt('AES_trainset.csv', delimiter=',', dtype=np.float, skiprows=1, usecols=range(1, 378))
X_test = np.loadtxt('AES_testset.csv', delimiter=',', dtype=np.float, skiprows=1, usecols=range(1, 378))

Y_train = np.loadtxt('AES_trainset.csv', delimiter=',', dtype=np.int, skiprows=1, usecols=380)
Y_test = np.loadtxt('AES_testset.csv', delimiter=',', dtype=np.int, skiprows=1, usecols=383)

Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)

test_inout = np.loadtxt('AES_testset.csv', delimiter=',', dtype=np.str, skiprows=1, usecols=0)

#Preprocessing
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf_nb = gnb.fit(X_train[:10000], Y_train[:10000])
Y_pred_gnb = clf_nb.predict(X_test)
cnf_matrix_gnb = confusion_matrix(Y_test, Y_pred_gnb)
print(cnf_matrix_gnb)
print(accuracy_score(Y_test, Y_pred_gnb))
target_names = ['HW0', 'HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'HW6', 'HW7', 'HW8']
print(classification_report(Y_test, Y_pred_gnb, target_names=target_names))
#SVM
from sklearn import svm
parameters = {'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1], 'gamma':[0.001, 0.01, 0.1, 1]}
SVM = svm.SVC()
clf_svm = GridSearchCV(SVM, parameters, cv=5, n_jobs=4, scoring='accuracy')
clf_svm.fit(X_train, Y_train)
print(clf_svm.best_estimator_)
print(clf_svm.best_params_)
print(clf_svm.best_score_)
Y_pred_svm = clf_svm.predict(X_test)
cnf_matrix_svm = confusion_matrix(Y_test, Y_pred_svm)
print(cnf_matrix_svm)
print(accuracy_score(Y_test, Y_pred_svm))
target_names = ['HW0', 'HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'HW6', 'HW7', 'HW8']
print(classification_report(Y_test, Y_pred_svm, target_names=target_names))
SVM = svm.SVC(probability=True)
Y_pred_svm = clf_svm.predict_proba(X_test)
#MLP
from sklearn.neural_network import MLPClassifier
parameters = {'solver':['adam', 'lbfgs', 'sgd'], 'activation':['tanh','relu'], 'hidden_layer_sizes':[(50, 40, 30, 25, 40),
                                                                                                     (50, 25, 50)]}
mlp = MLPClassifier(early_stopping=True, random_state=1)
clf_mlp = GridSearchCV(mlp, parameters, cv=5, n_jobs=4, scoring='accuracy')
clf_mlp.fit(X_train, Y_train)
print(clf_mlp.best_estimator_)
print(clf_mlp.best_params_)
print(clf_mlp.best_score_)
Y_pred_mlp = clf_mlp.predict(X_test)
cnf_matrix_mlp = confusion_matrix(Y_test, Y_pred_mlp)
print(cnf_matrix_mlp)
print(accuracy_score(Y_test, Y_pred_mlp))
target_names = ['HW0', 'HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'HW6', 'HW7', 'HW8']
print(classification_report(Y_test, Y_pred_mlp, target_names=target_names))
Y_pred_mlp = clf_mlp.predict_proba(X_test)

#RF
from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators':[10, 50, 100, 300]}
rf = RandomForestClassifier(random_state=1)
clf_rf = GridSearchCV(rf, parameters, cv=5, n_jobs=4, scoring='accuracy')
clf_rf.fit(X_train, Y_train)
print(clf_rf.best_estimator_)
print(clf_rf.best_params_)
print(clf_rf.best_score_)
Y_pred_rf = clf_rf.predict(X_test)
cnf_matrix_rf = confusion_matrix(Y_test, Y_pred_rf)
print(cnf_matrix_rf)
print(accuracy_score(Y_test, Y_pred_rf))
target_names = ['HW0', 'HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'HW6', 'HW7', 'HW8']
print(classification_report(Y_test, Y_pred_rf, target_names=target_names))

AES_Sbox = np.array(
    [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82, 0xC9,
     0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F,
     0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07,
     0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3,
     0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58,
     0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3,
     0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F,
     0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
     0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC,
     0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A,
     0xAE, 0x08, 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70,
     0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8, 0x98, 0x11,
     0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF, 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42,
     0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16])

hw = np.array([bin(x).count("1") for x in range(256)])


def key_rank(clf, inout_test, traces_test, trueKey, kByte=0):
    p = clf.predict_proba(traces_test)
    rank = np.zeros(inout_test.shape[0])
    prob_vector = np.zeros(256)

    for i, v in enumerate(inout_test):
        for kh in range(0, 256):
            hemw = hw[AES_Sbox[bytes.fromhex(v)[kByte] ^ kh]]
            prob_vector[kh] += p[i][hemw]
        df = pd.DataFrame({'prob': prob_vector})
        df = df.sort_values(['prob'], ascending=False)
        df = df.reset_index()
        df.rename(columns={'index': 'keyH'},inplace=True)
        rank[i] = df[df.keyH == int(trueKey, 16)].index.tolist()[0]
    return rank


rank = key_rank(clf_rf, test_inout, X_test, 'de', 0)

keypred = pd.DataFrame(rank)
plt.plot(keypred.index.values[0:9500], keypred[0][0:9500])
plt.xlabel('# of traces')
plt.ylabel('Rank')
plt.xticks(np.arange(0, 9500, step=500))
plt.show()

