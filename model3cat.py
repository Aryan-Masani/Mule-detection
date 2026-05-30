from preprocessing import X_train,y_train, X_train_res, y_train_res, X_test, y_test
from catboost import CatBoostClassifier
import pandas as pd 

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC', # Focus on AUC for imbalanced data
    scale_pos_weight= imbalance_ratio, # Crucial for 0.9% mule rate
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    
    eval_set=(X_test, y_test),
    early_stopping_rounds=50 # Stop if the AUC stops improving
)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print(f"ROC-AUC Score: {roc_auc_score(y_test, probs)}")
print(classification_report(y_test, preds))