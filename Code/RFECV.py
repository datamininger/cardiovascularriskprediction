from sklearn.metrics import roc_auc_score



def compute_auc(clf, X,y):
    """
    :param clf:
    :param X:
    :param y:
    :return:
    """
    y_pred = clf.predict(X)
    auc = roc_auc_score(y, y_pred)
    return  auc