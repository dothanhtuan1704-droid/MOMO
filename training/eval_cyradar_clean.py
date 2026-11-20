import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, r"e:\MOMO\training")
from train_phiusiil import extract_url_batch

cy = pd.read_parquet(r"e:\MOMO\data\external_test\cyradar_vn_0000.parquet")
is_url = cy["text"].astype(str).str.match(r"^https?://[^\s<>\"']{4,350}$")
cy_url = cy[is_url].copy().reset_index(drop=True)
print("CyRadar tong dong:", len(cy), "-> Loc rieng dong la URL that:", len(cy_url), "| dist:", cy_url["label"].value_counts().to_dict())

seen = set()
for fn, col in [("phiusiil_raw.parquet", "phiusiil_url"), ("mitake_urls.parquet", "url"), ("pirocheto_urls.parquet", "url")]:
    s = pd.read_parquet(os.path.join(r"e:\MOMO\data", fn), columns=[col])[col].astype(str).str.lower().str.replace(r"^https?://(www\.)?", "", regex=True).str.rstrip("/")
    seen.update(s.tolist())

norm_c = cy_url["text"].str.lower().str.replace(r"^https?://(www\.)?", "", regex=True).str.rstrip("/")
cy_new = cy_url[~norm_c.isin(seen)].drop_duplicates(subset=["text"]).reset_index(drop=True)
print("Sau khi loai URL trung voi Train -> Giu 100% URL MOI cua CyRadar VN:", len(cy_new), "| dist:", cy_new["label"].value_counts().to_dict())
print("Sample label=0:", cy_new[cy_new["label"] == 0]["text"].head(4).tolist())
print("Sample label=1:", cy_new[cy_new["label"] == 1]["text"].head(4).tolist())

b = joblib.load(r"e:\MOMO\models\phiusiil_bundle.joblib")
X_lex, norms, _, tlds = extract_url_batch(cy_new["text"].tolist())
X_lex["tld_phish_rate"] = [b["tld_rate_map"].get(t, b["global_rate"]) for t in tlds]
X_lex["tfidf_char_prob"] = b["char_clf"].predict_proba(b["char_vec"].transform(norms))[:, 1]
X_lex["tfidf_word_prob"] = b["word_clf"].predict_proba(b["word_vec"].transform(norms))[:, 1]
probs = b["lgb_model"].predict_proba(X_lex[b["feature_cols"]])[:, 1]
y = cy_new["label"].values
pred = (probs >= b["threshold"]).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"CYRADAR VN (PURE URL ONLY - 100% UNSEEN): ROC-AUC={roc_auc_score(y, probs):.4f} | PR-AUC={average_precision_score(y, probs):.4f} | F1={f1_score(y, pred):.4f} | Prec={precision_score(y, pred)*100:.2f}% | Rec={recall_score(y, pred)*100:.2f}% | FPR={fp/(fp+tn)*100:.2f}%")
print(f"CM: TN={tn} | FP={fp} | FN={fn} | TP={tp}")

