from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

def lr(conceptes, label):
    X_train, X_test, y_train, y_test = train_test_split(conceptes, label, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train_balanced, y_train_balanced)

    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)