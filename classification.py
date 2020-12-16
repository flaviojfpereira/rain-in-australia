from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



def linear_discriminant_analysis(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = LinearDiscriminantAnalysis()
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(model, x, y.ravel(), cv=kfold, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg

def nearest_centroid(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = NearestCentroid(metric='euclidean', shrink_threshold=None)  
    results = cross_val_score(model, x, y.ravel(), cv=kfold, scoring="accuracy")
    msg = 'accuracy: ' + str(results.mean()) +'\nBest_accuracy: ' + str(max(results)) + '\nstandard deviation: ' + str(results.std())
    print(msg)
    return msg

def random_forest(x, y):
    rfc = RandomForestClassifier(n_estimators=100)
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(rfc, x, y.ravel(), cv=10, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg

def decision_tree(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = DecisionTreeClassifier(max_depth=6)
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(model, x, y.ravel(), cv=kfold, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg

def gaussian_NB(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = GaussianNB()
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(model, x, y.ravel(), cv=kfold, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg

def k_neighbors(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = KNeighborsClassifier(n_neighbors=3)
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(model, x, y.ravel(), cv=kfold, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg

def svc(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = SVC(gamma='auto')
    scor = ['roc_auc', 'accuracy']
    results = cross_validate(model, x, y.ravel(), cv=kfold, scoring=scor)
    msg = 'roc_auc: ' + str(results["test_roc_auc"].mean()) + '\naccuracy: ' + str(results["test_accuracy"].mean()) + '\nBest_roc_auc: ' + str(max(results["test_roc_auc"])) + '\nbest_accuracy: ' + str(max(results["test_accuracy"])) + '\nroc_auc std: ' + str(results["test_roc_auc"].std()) + '\naccuracy std: ' + str(results["test_accuracy"].std())
    print("msg in:", msg)
    return msg
