import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
import joblib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_PARQUET_PATH = os.path.join(DATA_DIR, "paysim_raw.parquet")

# CHỈ DÙNG CÁC ĐẶC TRƯNG BIẾT TRƯỚC KHI DUYỆT GIAO DỊCH (STRICT PRE-TRANSACTION — ZERO LEAKAGE!)
# Đã loại bỏ hoàn toàn newbalanceOrig, newbalanceDest, error_balance_orig, error_balance_dest
FEATURE_NAMES_VI = {
    "type_code": "Loại giao dịch (TRANSFER / CASH_OUT / PAYMENT...)",
    "is_risky_type": "Nhóm giao dịch rủi ro cao (TRANSFER / CASH_OUT)",
    "amount": "Số tiền yêu cầu giao dịch",
    "amount_log": "Quy mô số tiền giao dịch (Log)",
    "oldbalanceOrg": "Số dư hiện tại của tài khoản gửi (trước GD)",
    "oldbal_org_log": "Quy mô số dư tài khoản gửi (Log)",
    "oldbalanceDest": "Số dư hiện tại của tài khoản nhận (trước GD)",
    "oldbal_dest_log": "Quy mô số dư tài khoản nhận (Log)",
    "amt_to_bal_ratio": "Tỷ lệ số tiền yêu cầu / số dư khả dụng (làm tròn 2 chữ số)",
    "projected_leftover_ratio": "Tỷ lệ số dư dự kiến còn lại sau giao dịch",
    "dest_zero_before": "Tài khoản nhận đang có số dư bằng 0 trước GD",
    "dest_is_merchant": "Tài khoản nhận là Merchant (M) hay Cá nhân (C)",
    "hour_of_day": "Khung giờ thực hiện giao dịch (0h - 23h)",
    "is_night_tx": "Giao dịch vào khung giờ đêm khuya (00h - 05h)",
    "amount_vs_type_median": "Tỷ lệ số tiền so với mức trung vị cùng loại GD",
    "dest_past_in_cnt": "Số lượt nhận tiền trước đó của tài khoản đích (Fan-in lịch sử)",
    "orig_past_out_cnt": "Số lượt gửi tiền trước đó của tài khoản nguồn",
    "dest_pagerank": "Chỉ số tập trung dòng tiền mạng lưới lịch sử (Graph PageRank)",
}
FEATURE_COLS = list(FEATURE_NAMES_VI.keys())
TYPE_MAP = {"PAYMENT": 0, "CASH_IN": 1, "DEBIT": 2, "CASH_OUT": 3, "TRANSFER": 4}


def build_pretx_features(df: pd.DataFrame, type_medians=None, pr=None):
    df = df.copy()
    df["type_code"] = df["type"].map(TYPE_MAP).fillna(0).astype(int)
    df["is_risky_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    df["hour_of_day"] = (df["step"] % 24).astype(int)
    df["is_night_tx"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    df["amount_log"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["oldbal_org_log"] = np.log1p(np.maximum(df["oldbalanceOrg"], 0.0))
    df["oldbal_dest_log"] = np.log1p(np.maximum(df["oldbalanceDest"], 0.0))

    # Làm tròn 2 chữ số thập phân để ngăn model lợi dụng sai số dấu phẩy động của trình mô phỏng
    df["amt_to_bal_ratio"] = np.clip(np.round(df["amount"] / (df["oldbalanceOrg"] + 1.0), 2), 0.0, 5.0)
    df["projected_leftover_ratio"] = np.clip(
        np.round((df["oldbalanceOrg"] - df["amount"]) / (df["oldbalanceOrg"] + 1.0), 2), -1.0, 1.0
    )
    df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

    if type_medians is None:
        type_medians = df.groupby("type")["amount"].median().to_dict()
    df["amount_vs_type_median"] = df["amount"] / (df["type"].map(type_medians).fillna(1.0) + 1.0)

    # Tần suất nhân quả (CHỈ đếm giao dịch quá khứ trước giao dịch hiện tại — cumcount)
    df["dest_past_in_cnt"] = df.groupby("nameDest").cumcount().astype(int)
    df["orig_past_out_cnt"] = df.groupby("nameOrig").cumcount().astype(int)

    if pr is None:
        risky_mask = (df["is_risky_type"] == 1) & ((df["dest_past_in_cnt"] >= 1) | (df["amount"] > 150_000))
        sub_df = df.loc[risky_mask, ["nameOrig", "nameDest", "amount"]]
        G = nx.DiGraph()
        for row in sub_df.itertuples(index=False):
            G.add_edge(row.nameOrig, row.nameDest, weight=float(row.amount))
        pr = nx.pagerank(G, alpha=0.85, max_iter=50) if G.number_of_nodes() > 0 else {}
        print(f"      -> Đồ thị NetworkX (Train quá khứ): {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    df["dest_pagerank"] = df["nameDest"].map(pr).fillna(0.0) * 1e5
    raw_graph_score = (
        np.clip(df["dest_past_in_cnt"] * 20.0, 0, 55)
        + np.clip(df["dest_pagerank"] * 12.0, 0, 30)
        + np.where((df["dest_zero_before"] == 1) & (df["amount"] > 100_000) & (df["is_risky_type"] == 1), 15.0, 0.0)
    )
    df["graph_risk_score"] = np.clip(raw_graph_score, 0, 100)
    return df, type_medians, pr


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    print("=" * 78)
    print("MODULE 1 (CLEAN STRICT PRE-TX): PAYSIM CHRONOLOGICAL FRAUD & GRAPH ENGINE")
    print("=" * 78)
    t_all = time.time()
    df = pd.read_parquet(RAW_PARQUET_PATH).sort_values("step").reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    train_raw, val_raw, test_raw = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]

    train_df, type_medians, pr = build_pretx_features(train_raw)
    val_df, _, _ = build_pretx_features(val_raw, type_medians, pr)
    test_df, _, _ = build_pretx_features(test_raw, type_medians, pr)

    X_train, y_train = train_df[FEATURE_COLS], train_df["isFraud"].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df["isFraud"].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df["isFraud"].astype(int)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=35,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )

    val_prob = lgb_model.predict_proba(X_val)[:, 1]
    best_th, val_f1 = find_best_threshold(y_val, val_prob)

    t_inf = time.time()
    test_prob = lgb_model.predict_proba(X_test)[:, 1]
    inf_ms = ((time.time() - t_inf) / len(X_test)) * 1000.0
    test_pred = (test_prob >= best_th).astype(int)

    roc_auc = roc_auc_score(y_test, test_prob)
    pr_auc = average_precision_score(y_test, test_prob)
    f1 = f1_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred)
    rec = recall_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)

    iso_model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        max_samples=4096,
        random_state=42,
        n_jobs=-1,
    )
    iso_model.fit(X_train[y_train == 0].sample(n=100_000, random_state=42))
    raw_iso_val = -iso_model.score_samples(X_val)
    iso_min, iso_max = float(np.percentile(raw_iso_val, 1)), float(np.percentile(raw_iso_val, 99.5))
    anom_test = np.clip((-iso_model.score_samples(X_test) - iso_min) / (iso_max - iso_min + 1e-9) * 100.0, 0, 100)

    print(f"KẾT QUẢ PAYSIM CLEAN PRE-TX TRÊN TẬP TEST TƯƠNG LAI ({len(X_test):,} GD | {int(y_test.sum())} gian lận):")
    print(f"  * Threshold (Val) : {best_th:.4f} (Val F1 = {val_f1:.4f})")
    print(f"  * ROC-AUC         : {roc_auc:.5f}")
    print(f"  * PR-AUC          : {pr_auc:.5f}")
    print(f"  * F1-Score        : {f1:.5f} (Prec={prec*100:.2f}% | Rec={rec*100:.2f}% | FPR={fpr*100:.4f}%)")
    print(f"  * Confusion Matrix: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print(f"  * Thời gian       : {time.time() - t_all:.1f}s")

    bundle = {
        "name": "PaySim Clean Pre-Transaction Fraud & Graph Engine",
        "lgb_model": lgb_model,
        "iso_model": iso_model,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "threshold": best_th,
        "feature_cols": FEATURE_COLS,
        "feature_names_vi": FEATURE_NAMES_VI,
        "type_map": TYPE_MAP,
        "type_medians": type_medians,
        "metrics": {
            "dataset": "PaySim (636,262 GD - Strict Pre-Transaction, bỏ hoàn toàn newbalanceOrig/Dest)",
            "split": "70% Train / 15% Val / 15% Test (Chronological - theo thời gian thực)",
            "test_size": int(len(X_test)),
            "test_frauds": int(y_test.sum()),
            "roc_auc": round(float(roc_auc), 5),
            "pr_auc": round(float(pr_auc), 5),
            "f1": round(float(f1), 5),
            "precision": round(float(prec), 5),
            "recall": round(float(rec), 5),
            "fpr": round(float(fpr), 6),
            "threshold": round(float(best_th), 4),
            "latency_ms": round(float(inf_ms), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "paysim_bundle.joblib"))

    test_df["p_sup"] = np.round(test_prob * 100.0, 2)
    test_df["s_anom"] = np.round(anom_test, 1)
    test_df["hybrid_score"] = np.round(
        np.clip(0.75 * test_df["p_sup"] + 0.15 * test_df["s_anom"] + 0.10 * test_df["graph_risk_score"], 0, 100), 1
    )
    sample_q = pd.concat([
        test_df[test_df["isFraud"] == 1].sort_values("hybrid_score", ascending=False).head(60),
        test_df[test_df["isFraud"] == 0].sort_values("hybrid_score", ascending=False).head(40),
        test_df[test_df["isFraud"] == 0].sample(n=100, random_state=42),
    ], ignore_index=True)
    sample_q.to_parquet(os.path.join(DATA_DIR, "paysim_ops_samples.parquet"), index=False)


if __name__ == "__main__":
    main()
