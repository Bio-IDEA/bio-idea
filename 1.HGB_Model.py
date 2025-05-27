import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

lable_path = 'label'
feature_path = 'feature'
node_features = []
node_tags = []
for ch in range(1, 23):
    lable = lable_path + '/chr' + str(ch) + '_label'
    feature = feature_path + '/chr' + str(ch) + '_feature'
    l = pd.read_csv(lable, sep='\t', header=None).values
    f = pd.read_csv(feature, sep='\t', header=None).values
    node_features.append(f)
    node_tags.append(l.flatten())

# import shap
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
auroc_scores = []
auprc_scores = []
myshap = []
for i in range(22):
    X_test = np.array(node_features[i])[:, 1:]
    y_test = np.array(node_tags[i].reshape(-1))
    X_train = np.vstack([node_features[j][:, 1:] for j in range(22) if j != i])
    y_train = np.concatenate([node_tags[j].reshape(-1) for j in range(22) if j != i])
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    auroc_scores.append(auroc)
    auprc_scores.append(auprc)
    print(f"Fold {i + 1}: AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
print(f"Average AUROC: {np.mean(auroc_scores):.4f}")
print(f"Average AUPRC: {np.mean(auprc_scores):.4f}")

