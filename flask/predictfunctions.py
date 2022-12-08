from preprocessfunctions import preprocesswithoutstem,clean_text,remove_stopwords1
from sklearn.svm import LinearSVC

def linearsvc(clf,summary):
    y_pred=clf.predict([summary])
    return {'reply':list(y_pred)}

def sgd(clf,summary):
    y_pred=clf.predict([summary])
    return {'reply':list(y_pred)}