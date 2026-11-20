import os
import time
import urllib.request
import numpy as np
import pandas as pd
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

SPARKOV_URL = (
    "https://huggingface.co/datasets/santosh3110/credit_card_fraud_transactions/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
SPARKOV_PATH = os.path.join(DATA_DIR, "sparkov_raw.parquet")

FEATURE_NAMES_VI = {
    "amt": "Số tiền thanh toán (USD)",
    "amt_log": "Quy mô số tiền thanh toán (Log)",
    "category_code": "Danh mục mua sắm (Shopping Net / Grocery / Travel...)",
    "hour": "Giờ thanh toán trong ngày (0h - 23h)",
    "hour_sin": "Chu kỳ giờ giao dịch (Sin)",
    "hour_cos": "Chu kỳ giờ giao dịch (Cos)",
    "is_night": "Thanh toán vào khung giờ đêm khuya (22h - 04h)",
    "day_of_week": "Ngày trong tuần",
    "is_weekend": "Giao dịch vào cuối tuần",
    "cardholder_age": "Tuổi của chủ thẻ",
    "gender_code": "Giới tính chủ thẻ",
    "city_pop_log": "Quy mô dân số thành phố cư trú (Log)",
    "distance_km": "Khoảng cách địa lý từ nhà chủ thẻ tới cửa hàng (km)",
    "time_since_last_tx_sec": "Thời gian từ lần quẹt thẻ gần nhất (giây)",
    "time_since_last_2tx_sec": "Thời gian của 2 giao dịch gần nhất (nhịp quẹt liên tiếp)",
    "dist_from_last_merch_km": "Khoảng cách từ điểm thanh toán trước đó (km)",
    "travel_speed_kmh": "Tốc độ di chuyển giữa 2 lần quẹt thẻ (Impossible Travel - km/h)",
    "tx_count_last_1h": "Số lần quẹt thẻ dồn dập trong 1 giờ gần nhất",
    "tx_count_last_24h": "Số lần quẹt thẻ trong 24 giờ gần nhất",
    "amt_sum_last_24h": "Tổng số tiền chi tiêu trong 24 giờ gần nhất",
    "amt_vs_prev_ratio": "Tỷ lệ số tiền so với giao dịch liền trước",
    "same_cat_as_prev": "Quẹt liên tiếp cùng danh mục mua sắm",
    "user_tx_count_prev": "Tổng số giao dịch lịch sử trước đó của chủ thẻ",
    "amt_ratio_user_mean": "Số tiền gấp bao nhiêu lần mức chi tiêu TB của chính chủ thẻ",
    "amt_diff_user_mean": "Độ chênh lệch so với mức chi tiêu TB của chủ thẻ",
    "amt_zscore_cat": "Độ lệch chuẩn (Z-Score) số tiền trong cùng danh mục mua sắm",
    "amt_ratio_cat_median": "Tỷ lệ số tiền so với trung vị danh mục mua sắm",
    "amt_ratio_user_cat_mean": "Tỷ lệ số tiền so với mức chi TB của chủ thẻ ở riêng danh mục này",
    "cat_fraud_rate_train": "Mức độ rủi ro lịch sử của danh mục mua sắm",
    "merch_fraud_rate_train": "Mức độ rủi ro lịch sử của cửa hàng (Merchant)",
}
FEATURE_COLS = list(FEATURE_NAMES_VI.keys())


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def download_sparkov():
    if os.path.exists(SPARKOV_PATH) and os.path.getsize(SPARKOV_PATH) > 50_000_000:
        print(f"[1/4] Đã có sẵn bộ dữ liệu Sparkov tại: {SPARKOV_PATH}")
        return
    print("[1/4] Đang tải bộ dữ liệu chuẩn Sparkov Credit Card Fraud (~154 MB) từ Hugging Face Hub...")
    t0 = time.time()
    urllib.request.urlretrieve(SPARKOV_URL, SPARKOV_PATH)
    print(f"      -> Tải hoàn tất ({os.path.getsize(SPARKOV_PATH)/(1024*1024):.1f} MB) trong {time.time()-t0:.1f}s!")


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def compute_rolling_velocity(df: pd.DataFrame):
    """Tính số giao dịch trong 1h, 24h và tổng tiền 24h theo từng chủ thẻ (chỉ nhìn về quá khứ - O(N))."""
    n = len(df)
    cnt_1h = np.zeros(n, dtype=np.float32)
    cnt_24h = np.zeros(n, dtype=np.float32)
    sum_24h = np.zeros(n, dtype=np.float32)

    cc_vals = df["cc_num"].values
    ts_vals = df["unix_time"].values
    amt_vals = df["amt"].values

    # Nhóm theo cc_num nhưng giữ nguyên thứ tự thời gian (df đã sort theo unix_time)
    order = np.argsort(cc_vals, kind="mergesort")
    cc_s = cc_vals[order]
    ts_s = ts_vals[order]
    amt_s = amt_vals[order]

    # Tìm biên của từng cc_num
    change_idx = np.flatnonzero(np.r_[True, cc_s[1:] != cc_s[:-1], True])
    for k in range(len(change_idx) - 1):
        start, end = change_idx[k], change_idx[k + 1]
        t_arr = ts_s[start:end]
        a_arr = amt_s[start:end]
        m = end - start
        c1 = np.zeros(m, dtype=np.float32)
        c24 = np.zeros(m, dtype=np.float32)
        s24 = np.zeros(m, dtype=np.float32)
        p1 = 0
        p24 = 0
        run_sum = 0.0
        for i in range(m):
            ti = t_arr[i]
            while p1 < i and ti - t_arr[p1] > 3600:
                p1 += 1
            while p24 < i and ti - t_arr[p24] > 86400:
                run_sum -= a_arr[p24]
                p24 += 1
            c1[i] = i - p1
            c24[i] = i - p24
            s24[i] = run_sum
            run_sum += a_arr[i]
        orig_indices = order[start:end]
        cnt_1h[orig_indices] = c1
        cnt_24h[orig_indices] = c24
        sum_24h[orig_indices] = s24
    return cnt_1h, cnt_24h, sum_24h


def main():
    print("=" * 75)
    print("MODULE 2 (NÂNG CẤP): SPARKOV PAYMENT & SHOPPING FRAUD ENGINE (VELOCITY + GEO)")
    print("=" * 75)
    t_all = time.time()
    download_sparkov()

    df = pd.read_parquet(SPARKOV_PATH)
    print(f"      -> Tổng số dòng Sparkov: {len(df):,} giao dịch | Gian lận: {int(df['is_fraud'].sum()):,} ({df['is_fraud'].mean()*100:.4f}%)")

    print("[2/4] Sắp xếp theo thời gian thực & trích xuất 30 đặc trưng hành vi nhân quả + Velocity 1h/24h...")
    df = df.sort_values("unix_time").reset_index(drop=True)

    dt = pd.to_datetime(df["trans_date_trans_time"], format="mixed")
    dob = pd.to_datetime(df["dob"], format="mixed")
    df["hour"] = dt.dt.hour.astype(int)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    df["day_of_week"] = dt.dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["cardholder_age"] = np.clip((dt - dob).dt.days / 365.25, 18, 100)
    df["gender_code"] = (df["gender"] == "M").astype(int)
    df["city_pop_log"] = np.log1p(np.maximum(df["city_pop"].astype(float), 0.0))
    df["amt_log"] = np.log1p(np.maximum(df["amt"].astype(float), 0.0))

    categories = sorted(df["category"].unique().tolist())
    cat_map = {c: i for i, c in enumerate(categories)}
    df["category_code"] = df["category"].map(cat_map).astype(int)

    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Đặc trưng chuỗi hành vi nhân quả quá khứ (shift >= 1)
    grp = df.groupby("cc_num", sort=False)
    prev_time = grp["unix_time"].shift(1)
    prev2_time = grp["unix_time"].shift(2)
    prev_lat = grp["merch_lat"].shift(1)
    prev_long = grp["merch_long"].shift(1)
    prev_amt = grp["amt"].shift(1)
    prev_cat = grp["category_code"].shift(1)

    df["time_since_last_tx_sec"] = (df["unix_time"] - prev_time).fillna(86400.0).clip(lower=1.0)
    df["time_since_last_2tx_sec"] = (df["unix_time"] - prev2_time).fillna(172800.0).clip(lower=1.0)
    dist_prev = haversine_km(
        prev_lat.fillna(df["lat"]), prev_long.fillna(df["long"]), df["merch_lat"], df["merch_long"]
    )
    df["dist_from_last_merch_km"] = np.where(prev_lat.isna(), df["distance_km"], dist_prev)
    df["travel_speed_kmh"] = np.clip(
        df["dist_from_last_merch_km"] / (df["time_since_last_tx_sec"] / 3600.0 + 1e-4), 0.0, 5000.0
    )
    df["amt_vs_prev_ratio"] = np.clip(df["amt"] / (prev_amt.fillna(df["amt"]) + 1.0), 0.0, 100.0)
    df["same_cat_as_prev"] = (df["category_code"] == prev_cat).astype(int)

    # Rolling Velocity 1h & 24h (chỉ nhìn quá khứ)
    c1h, c24h, s24h = compute_rolling_velocity(df)
    df["tx_count_last_1h"] = c1h
    df["tx_count_last_24h"] = c24h
    df["amt_sum_last_24h"] = s24h

    # Tỷ lệ chi tiêu so với trung bình lịch sử toàn phần & theo riêng từng danh mục của chủ thẻ
    df["user_tx_count_prev"] = grp.cumcount()
    cum_amt_prev = grp["amt"].cumsum() - df["amt"]
    user_prev_mean = np.where(df["user_tx_count_prev"] > 0, cum_amt_prev / np.maximum(df["user_tx_count_prev"], 1), df["amt"])
    df["amt_ratio_user_mean"] = np.clip(df["amt"] / (user_prev_mean + 1.0), 0.0, 100.0)
    df["amt_diff_user_mean"] = df["amt"] - user_prev_mean

    grp_uc = df.groupby(["cc_num", "category_code"], sort=False)
    uc_cnt = grp_uc.cumcount()
    uc_sum = grp_uc["amt"].cumsum() - df["amt"]
    uc_mean = np.where(uc_cnt > 0, uc_sum / np.maximum(uc_cnt, 1), user_prev_mean)
    df["amt_ratio_user_cat_mean"] = np.clip(df["amt"] / (uc_mean + 1.0), 0.0, 100.0)

    # Chronological Split: 70% Train / 15% Validation / 15% Test
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()

    global_rate = float(train_df["is_fraud"].mean())
    cat_stats = train_df.groupby("category")["amt"].agg(["mean", "std", "median"]).to_dict("index")
    cat_fraud_agg = train_df.groupby("category")["is_fraud"].agg(["count", "mean"])
    cat_fraud_rate = ((cat_fraud_agg["count"] * cat_fraud_agg["mean"] + 50 * global_rate) / (cat_fraud_agg["count"] + 50)).to_dict()

    merch_agg = train_df.groupby("merchant")["is_fraud"].agg(["count", "mean"])
    merch_fraud_rate = ((merch_agg["count"] * merch_agg["mean"] + 30 * global_rate) / (merch_agg["count"] + 30)).to_dict()

    for part in (train_df, val_df, test_df):
        c_mean = part["category"].map(lambda c: cat_stats.get(c, {"mean": 70.0})["mean"])
        c_std = part["category"].map(lambda c: max(cat_stats.get(c, {"std": 50.0})["std"], 1.0))
        c_med = part["category"].map(lambda c: cat_stats.get(c, {"median": 50.0})["median"])
        part["amt_zscore_cat"] = (part["amt"] - c_mean) / c_std
        part["amt_ratio_cat_median"] = part["amt"] / (c_med + 1.0)
        part["cat_fraud_rate_train"] = part["category"].map(cat_fraud_rate).fillna(global_rate)
        part["merch_fraud_rate_train"] = part["merchant"].map(merch_fraud_rate).fillna(global_rate)

    X_train, y_train = train_df[FEATURE_COLS], train_df["is_fraud"].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df["is_fraud"].astype(int)
    X_test, y_test = test_df[FEATURE_COLS], test_df["is_fraud"].astype(int)

    print(f"[3/4] Huấn luyện LightGBM (Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,})...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2200,
        learning_rate=0.025,
        num_leaves=95,
        max_depth=10,
        min_child_samples=35,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
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
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
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

    print("[4/4] Huấn luyện Isolation Forest cho hành vi mua sắm dị biệt...")
    iso_model = IsolationForest(
        n_estimators=120,
        contamination=0.02,
        max_samples=4096,
        random_state=42,
        n_jobs=-1,
    )
    iso_model.fit(X_train[y_train == 0].sample(n=100_000, random_state=42))
    raw_iso_val = -iso_model.score_samples(X_val)
    iso_min, iso_max = float(np.percentile(raw_iso_val, 1)), float(np.percentile(raw_iso_val, 99.5))

    print("\n" + "-" * 75)
    print(f"KẾT QUẢ SPARKOV (NÂNG CẤP) TRÊN TẬP TEST TƯƠNG LAI ({len(X_test):,} giao dịch | {int(y_test.sum())} gian lận):")
    print(f"  * Best Threshold (từ Val) : {best_th:.4f} (Val F1 = {val_f1:.4f})")
    print(f"  * ROC-AUC                 : {roc_auc:.5f}")
    print(f"  * PR-AUC (Avg Precision)  : {pr_auc:.5f}")
    print(f"  * F1-Score                : {f1:.5f}")
    print(f"  * Precision (Độ chuẩn xác): {prec*100:.2f}%")
    print(f"  * Recall (Tỷ lệ bắt đúng) : {rec*100:.2f}%")
    print(f"  * False Positive Rate     : {fpr*100:.4f}% (Chỉ {fp} cảnh báo nhầm / {tn+fp:,} giao dịch thật)")
    print(f"  * Confusion Matrix        : TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print(f"  * Tổng thời gian chạy     : {time.time() - t_all:.1f}s")
    print("-" * 75)

    bundle = {
        "name": "Sparkov Credit Card Payment & Shopping Fraud Engine",
        "lgb_model": lgb_model,
        "iso_model": iso_model,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "threshold": best_th,
        "feature_cols": FEATURE_COLS,
        "feature_names_vi": FEATURE_NAMES_VI,
        "cat_map": cat_map,
        "cat_stats": cat_stats,
        "cat_fraud_rate": cat_fraud_rate,
        "global_rate": global_rate,
        "metrics": {
            "dataset": f"Sparkov Credit Card Fraud ({len(df):,} giao dịch)",
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
    joblib.dump(bundle, os.path.join(MODEL_DIR, "sparkov_bundle.joblib"))

    test_df["risk_prob"] = np.round(test_prob * 100.0, 2)
    sample_sp = pd.concat([
        test_df[test_df["is_fraud"] == 1].sort_values("risk_prob", ascending=False).head(60),
        test_df[test_df["is_fraud"] == 0].sort_values("risk_prob", ascending=False).head(40),
        test_df[test_df["is_fraud"] == 0].sample(n=100, random_state=42),
    ], ignore_index=True)
    meta_cols = [
        "trans_date_trans_time", "merchant", "category", "city", "state",
        "lat", "long", "merch_lat", "merch_long", "is_fraud", "risk_prob"
    ]
    keep_cols = meta_cols + [c for c in FEATURE_COLS if c not in meta_cols]
    sample_sp[keep_cols].to_parquet(os.path.join(DATA_DIR, "sparkov_ops_samples.parquet"), index=False)


if __name__ == "__main__":
    main()
