from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    parameters = {'max_depth': range(1, X.shape[1] + 1)}
    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='auc', use_label_encoder=False)
    grids = GridSearchCV(xgb_clf, parameters, scoring='accuracy', cv=10)
    grids.fit(X_train, y_train)

    depth = grids.best_params_['max_depth']

    xgb_clf = XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                  max_depth=depth, use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)
    acc_decision_tree = accuracy_score(y_test, xgb_clf.predict(X_test))
    
    return acc_decision_tree, depth

def train_SVM(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = scale(X_train)
    X_test = scale(X_test)

    svm_clf = SVC(random_state=1)
    svm_clf.fit(X_train, y_train)

    acc_svm = accuracy_score(y_test, svm_clf.predict(X_test))
    return acc_svm

def train_randomforest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    parameters = {'max_depth': range(1, X.shape[1] + 1)}

    rf_clf = RandomForestClassifier(random_state=1)
    grids = GridSearchCV(rf_clf, parameters, scoring='accuracy', cv=10)
    grids.fit(X_train, y_train)

    depth = grids.best_params_['max_depth']

    rf_clf = RandomForestClassifier(max_depth=depth, random_state=1)
    rf_clf.fit(X, y)
    acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))
    return acc_rf, depth

def train_cart(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    parameters = {'max_depth': range(1, X.shape[1] + 1)}

    dt_clf = DecisionTreeClassifier(random_state=1)
    grids = GridSearchCV(dt_clf, parameters, scoring='accuracy', cv=10)
    grids.fit(X_train, y_train)
    
    depth = grids.best_params_['max_depth']

    rf_clf = DecisionTreeClassifier(max_depth=depth, random_state=1)
    rf_clf.fit(X, y)
    acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))
    return acc_rf, depth