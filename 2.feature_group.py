import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import combinations, chain

# 路径设置
lable_path = './label'
feature_path = 'feature'
output_dir = 'your_path'


os.makedirs(output_dir, exist_ok=True)


node_features = []
node_tags = []
for ch in range(1, 23):
    label_file = os.path.join(lable_path, f'chr{ch}_label')
    feature_file = os.path.join(feature_path, f'chr{ch}_feature')
    l = pd.read_csv(label_file, sep='\t', header=None).values
    f = pd.read_csv(feature_file, sep='\t', header=None).values
    node_features.append(f)
    node_tags.append(l.flatten())


feature_cols = {
    'RnaSeq': 0,
    'CTCF': 1,
    'DNase': 2,
    'H3k27ac': 3,
    'H3k4me3': 4,
    'TCI_Qnor': 5
}


all_features = list(feature_cols.keys())


feature_combinations = list(chain.from_iterable(combinations(all_features, r) for r in range(1, len(all_features) + 1)))

all_results = []


for features in feature_combinations:
    print(features)
    feature_indices = [feature_cols[f] for f in features]
    auroc_scores = []
    auprc_scores = []


    for i in range(22):

        X_test = np.array(node_features[i])[:, feature_indices]
        y_test = np.array(node_tags[i].reshape(-1))

        X_train = np.vstack([node_features[j][:, feature_indices] for j in range(22) if j != i])
        y_train = np.concatenate([node_tags[j].reshape(-1) for j in range(22) if j != i])


        model = HistGradientBoostingClassifier()

        # 训练模型
        model.fit(X_train, y_train)

        # 预测测试集
        y_pred_proba = model.predict_proba(X_test)[:, 1]


        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)

        print(f"Fold {i + 1}: AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    avg_auroc = np.mean(auroc_scores)
    avg_auprc = np.mean(auprc_scores)
    all_results.append({'features': '_'.join(features), 'avg_auroc': avg_auroc, 'avg_auprc': avg_auprc})


    result_df = pd.DataFrame({
        'AUROC': auroc_scores,
        'AUPRC': auprc_scores
    })
    result_df.to_csv(os.path.join(output_dir, f"{('_').join(features)}_results.csv"), index=False)


summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)