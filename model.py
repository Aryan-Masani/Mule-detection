from preprocessing import X_train,y_train, X_train_res, y_train_res, X_test, y_test
from xgboost import XGBClassifier
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=99
)

"""model.fit(X_train_res,y_train_res)

#y_pred = model.predict(X_test)

# Instead of y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_new = (y_probs > 0.08).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_new))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_new))"""


# without SMOTE

model.fit(X_train,y_train)

#y_pred = model.predict(X_test)

# Instead of y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_new = (y_probs > 0.08).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_new))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_new))


"""Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.91      0.95      1801
           1       0.06      0.62      0.10        16

    accuracy                           0.90      1817
   macro avg       0.53      0.77      0.53      1817
weighted avg       0.99      0.90      0.94      1817

Confusion Matrix:
[[1634  167]
 [   6   10]]"""


