import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def best_th_eval(y_va, p_va, y_te, p_te):
    pr, rc, th = precision_recall_curve(y_va, p_va)
    bth = float(th[np.argmax(2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12))])
    pred = (p_te >= bth).astype(int)
    return {
        "th": bth,
        "roc_auc": roc_auc_score(y_te, p_te),
        "pr_auc": average_precision_score(y_te, p_te),
        "f1": f1_score(y_te, pred),
        "prec": precision_score(y_te, pred, zero_division=0),
        "rec": recall_score(y_te, pred, zero_division=0),
    }


def audit_paysim_strict():
    df = pd.read_parquet(os.path.join(DATA, "paysim_raw.parquet")).sort_values("step").reset_index(drop=True)
    type_map = {"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4}
    df["type_code"] = df["type"].map(type_map)
    df["is_risky_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    df["oldbal_org_log"] = np.log1p(df["oldbalanceOrg"])
    df["oldbal_dest_log"] = np.log1p(df["oldbalanceDest"])
    # Tỷ lệ làm tròn 2 chữ số (chống lợi dụng lỗi dấu phẩy động của trình giả lập)
    df["amt_to_bal_ratio"] = np.clip(np.round(df["amount"] / (df["oldbalanceOrg"] + 1.0), 2), 0.0, 5.0)
    df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(int)
    df["dest_past_in_cnt"] = df.groupby("nameDest").cumcount()
    df["orig_past_out_cnt"] = df.groupby("nameOrig").cumcount()
    df["projected_leftover_ratio"] = np.clip((df["oldbalanceOrg"] - df["amount"]) / (df["oldbalanceOrg"] + 1.0), -1.0, 1.0)
    df["projected_leftover_ratio"] = np.round(df["projected_leftover_ratio"], 2)

    cols = [
        "type_code", "is_risky_type", "hour", "is_night", "amount_log",
        "oldbal_org_log", "oldbal_dest_log", "amt_to_bal_ratio",
        "projected_leftover_ratio", "dest_zero_before", "dest_is_merchant",
        "dest_past_in_cnt", "orig_past_out_cnt",
    ]
    n = len(df)
    tr, va, te = df.iloc[:int(n * 0.70)], df.iloc[int(n * 0.70):int(n * 0.85)], df.iloc[int(n * 0.85):]
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(
        tr[cols], tr["isFraud"],
        eval_set=[(va[cols], va["isFraud"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    res = best_th_eval(va["isFraud"], m.predict_proba(va[cols])[:, 1], te["isFraud"], m.predict_proba(te[cols])[:, 1])
    print("=" * 78)
    print("AUDIT 1: PAYSIM STRICT PRE-TRANSACTION + CHRONOLOGICAL (BỎ HẲN newbalanceOrig/Dest)")
    print(f"  Test={len(te):,} | Frauds={int(te['isFraud'].sum())} | ROC-AUC={res['roc_auc']:.4f} | PR-AUC={res['pr_auc']:.4f} | F1={res['f1']:.4f} | Prec={res['prec']*100:.1f}% | Rec={res['rec']*100:.1f}%")
    imp = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=cols)
    print("  Top gain (%):", (imp / imp.sum() * 100).sort_values(ascending=False).head(6).round(1).to_dict())


def audit_sparkov_strict():
    df = pd.read_parquet(os.path.join(DATA, "sparkov_raw.parquet")).sort_values("unix_time").reset_index(drop=True)
    # Test Sparkov trên TẬP CHỦ THẺ HOÀN TOÀN MỚI (GroupShuffleSplit theo cc_num) + bỏ merch_fraud_rate!
    dt = pd.to_datetime(df["trans_date_trans_time"], format="mixed")
    dob = pd.to_datetime(df["dob"], format="mixed")
    df["hour"] = dt.dt.hour.astype(int)
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    df["cardholder_age"] = np.clip((dt - dob).dt.days / 365.25, 18, 100)
    df["amt_log"] = np.log1p(df["amt"])
    df["category_code"] = df["category"].astype("category").cat.codes

    # Kiểm tra đánh giá KHÔNG có burst history (chỉ xét giao dịch đơn lẻ đầu tiên!) vs CÓ burst history
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_i, tmp_i = next(gss.split(df, df["is_fraud"], groups=df["cc_num"]))
    tmp = df.iloc[tmp_i].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    va_r, te_r = next(gss2.split(tmp, tmp["is_fraud"], groups=tmp["cc_num"]))
    va_i, te_i = tmp_i[va_r], tmp_i[te_r]

    # Single-tx only (không dùng lịch sử 24h) trên chủ thẻ hoàn toàn mới
    cols_single = ["amt", "amt_log", "category_code", "hour", "is_night", "cardholder_age", "distance_km"]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(
        df.iloc[tr_i][cols_single], df.iloc[tr_i]["is_fraud"],
        eval_set=[(df.iloc[va_i][cols_single], df.iloc[va_i]["is_fraud"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    res = best_th_eval(
        df.iloc[va_i]["is_fraud"], m.predict_proba(df.iloc[va_i][cols_single])[:, 1],
        df.iloc[te_i]["is_fraud"], m.predict_proba(df.iloc[te_i][cols_single])[:, 1],
    )
    print("=" * 78)
    print("AUDIT 2: SPARKOV TRÊN CHỦ THẺ HOÀN TOÀN MỚI (UNSEEN CARDHOLDERS, KHÔNG DÙNG LỊCH SỬ 24H)")
    print(f"  Test={len(te_i):,} | Frauds={int(df.iloc[te_i]['is_fraud'].sum())} | ROC-AUC={res['roc_auc']:.4f} | PR-AUC={res['pr_auc']:.4f} | F1={res['f1']:.4f} | Prec={res['prec']*100:.1f}% | Rec={res['rec']*100:.1f}%")


if __name__ == "__main__":
    audit_paysim_strict()
    audit_sparkov_strict()

