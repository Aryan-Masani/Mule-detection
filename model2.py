from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import X_train,y_train, X_train_res, y_train_res, X_test, y_test

rf_model = RandomForestClassifier(n_estimators=200,
                                  max_depth=10,
                                  class_weight='balanced_subsample'
                                  ,random_state=42)

rf_model.fit(X_train_res, y_train_res)

# 3. Use the same 0.08 threshold you used for XGBoost to compare them fairly
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_pred = (rf_probs > 0.30).astype(int)

print(classification_report(y_test, rf_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))