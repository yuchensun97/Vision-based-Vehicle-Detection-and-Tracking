import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import *
from sklearn.model_selection import train_test_split
import os
import time
import pickle

from constant import *


def train(X, y):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    model = LinearSVC(max_iter=MAX_ITER)

    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()

    print("{} seconds to train".format(round(t2 - t1, 2)))
    print("Test Accuracy = {}".format(round(model.score(X_test, y_test), 4)))

    # Save model and scaler
    pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))


def test(X, y):
    model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))

    X = scaler.transform(X)
    print("Test Accuracy = {}".format(round(model.score(X, y), 4)))
