
import A_load_data
from konlpy.tag import Hannanum
import fasttext
import numpy as np

from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score,f1_score
import random


hannanum = Hannanum()
ft = fasttext.load_model("Fasttext/cc.ko.300.bin")

randomseed = 3407

acclistsvm = []
f1listsvm = []

acclistknn3 = []
f1listknn3 = []

for iter in range(5):

    np.random.seed(randomseed + iter)
    random.seed(randomseed + iter)

    X_train, Y_train = A_load_data.load_weibo_train()
    X_test, Y_test = A_load_data.load_weibo_test()
    sentence_vectors_train = []
    for sentence in X_train:
        words = hannanum.morphs(sentence)
        vectors = []
        for word in words:
            id = ft.get_word_vector(word)
            vectors.append(id)
        vectors = np.vstack(vectors)
        mean = np.mean(vectors, axis=0)
        sentence_vectors_train.append(mean)

    X_train = np.vstack(sentence_vectors_train)
    sentence_vectors_test = []
    for sentence in X_test:
        words = hannanum.morphs(sentence)
        vectors = []
        for word in words:
            id = ft.get_word_vector(word)
            vectors.append(id)
        vectors = np.vstack(vectors)
        mean = np.mean(vectors, axis=0)
        sentence_vectors_test.append(mean)
    X_test = np.vstack(sentence_vectors_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    print("SVM")
    print(accuracy_score(Y_test, preds))
    acclistsvm.append(accuracy_score(Y_test, preds))
    f1listsvm.append(f1_score(Y_test, preds, average="macro"))

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    preds = knn.predict(X_test)
    print("KNN -3")
    print(accuracy_score(Y_test, preds))
    acclistknn3.append(accuracy_score(Y_test, preds))
    f1listknn3.append(f1_score(Y_test, preds, average="macro"))

acclistsvm = np.array(acclistsvm)
f1listsvm = np.array(f1listsvm)
print("svm")
print(np.mean(acclistsvm), np.mean(f1listsvm))

acclistknn3 = np.array(acclistknn3)
f1listknn3 = np.array(f1listknn3)
print("knn-3")
print(np.mean(acclistknn3), np.mean(f1listknn3))

