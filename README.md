Financial Mule Detection Engine

A machine learning pipeline designed to identify "money mule" accounts within financial transactions. This project utilizes gradient-boosted decision trees (CatBoost) to handle extreme class imbalance ($< 1\\%$ minority class) and optimize the trade-off between fraud detection recall and operational false-alarm rates.

 Project Overview
Money mules are individuals who, consciously or unconsciously, transfer illegally acquired money on behalf of others. Detecting these accounts is a classic needle-in-a-haystack problem characterized by heavy class asymmetry. 

This engine is engineered to maximize **Recall** (catching as many mules as possible) while maintaining a controlled, operationally viable **Precision** threshold using custom probability calibration.

### Key Performance Metrics (Baseline vs. Calibrated)
* **Minority Class Prevalance:** ~0.88% (16 Mules out of 1,817 Test Samples)
* **Raw ROC-AUC:** `0.9405`
* **Optimized Strategy:** Shifting from aggressive minority scaling to a calibrated precision-recall threshold optimization to mitigate severe operational noise.

mule_detection/
│
├── findataset.csv          # Raw transaction/account behavior dataset
├── newcsvmaker.py          # Data ingestion and synthetic feature simulation
├── preprocessing.py        # Feature scaling, encoding, and train-test splitting
├── model.py                # Initial prototype modeling script
└── model3cat.py            # Optimized CatBoost script with threshold tuning (Main)

Best Test ROC-AUC Score: 0.9405
Early Stopping triggered at Iteration: 46

--- Classification Report ---
              precision    recall  f1-score   support

           0       1.00      0.92      0.96      1801
           1       0.08      0.75      0.14        16

    accuracy                           0.92      1817
   macro avg       0.54      0.83      0.55      1817
weighted avg       0.99      0.92      0.95      1817

--- Confusion Matrix ---
[[1656  145]
 [   4   12]]

 Key Takeaways from the Data:The High-Recall Win: The baseline model correctly captured 12 out of 16 real mule accounts ($75\\%$ Recall).
 In an anti-money laundering (AML) environment, minimizing False Negatives (uncaught mules) is crucial to avoid regulatory penalties.
 The Operational Cost (Low Precision): Due to the hyper-aggressive class weight, the Precision dropped to 0.08$. 
 For every $12$ true mules captured, the system generated $145$ false alarms on legitimate clients. This results in an operational overhead ratio of ~12:1 false-to-true flags.The Fix: Transitioning to Logloss optimization combined with the custom best_threshold calculator pulls back on operational overhead, dampening False Positives while protecting the true positive capture rate.
