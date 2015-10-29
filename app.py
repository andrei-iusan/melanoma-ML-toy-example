from flask import Flask, render_template
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.metrics import precision_recall_fscore_support
precision_recall = precision_recall_fscore_support

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

alg_refs = ["KNN", "LinSVM", "RBFSVM", "DecTree", "RandForest", "AdaBoost",
            "NaiveBayes", "LDA", "QDA"]
algs = zip(names, alg_refs, classifiers)
Algorithm = namedtuple("Algorithm", ["caption", "href", "classifier"])
Algorithms = [Algorithm(name, href, classifier) for name, href, classifier in algs]

DEBUG = True
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', algorithms=Algorithms)

if __name__ == '__main__':
    app.run(debug=DEBUG)
