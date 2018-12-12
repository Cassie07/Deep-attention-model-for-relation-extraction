import pickle
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


def LR_model(X_train, y_train, k_fold):
    """

    :param train: X_train is a csr_matrix
    :param test: y_train is a nunmpy array
    :param output: the path to save the model parameter
    :return:
    """

    LR_model = LogisticRegression(penalty='l2', C=0.3, solver='lbfgs', class_weight='balanced')
#    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(LR_model, X_train, y_train, cv=k_fold)
    print scores



if __name__ == '__main__':
    train_path = os.path.abspath('..')+'/data/LR/train'
    test_dataset = os.path.abspath('..')+'/data/LR/test_dataset'
    model_output = os.path.abspath('..')+'/data/LR/parameter'

    with open(train_path+'/X.pkl', 'r') as xpkl:
        X_train = pickle.load(xpkl)
    with open(train_path+'/y.pkl', 'r') as ypkl:
        y_train = pickle.load(ypkl)
    LR_model(X_train, y_train, k_fold=3)


