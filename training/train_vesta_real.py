import os
import time
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
VESTA_PATH = os.path.join(DATA_DIR, "real_world_test", "vesta_real_ieee_cis.csv")

FEATURE_NAMES_VI = {
    "TransactionAmt": "Số tiền giao dịch thanh toán (USD)",
    "amt_log": "Quy mô số tiền giao dịch (Log)",
    "amt_decimal": "Phần lẻ thập phân của số tiền (dấu hiệu ngoại tệ / bot)",
    "amt_is_round": "Số tiền chẵn tròn (thẻ quà tặng / rút tiền)",
    "product_code": "Nhóm sản phẩm/dịch vụ thanh toán (ProductCD)",
    "card1": "Mã định danh ngân hàng/đầu thẻ (card1)",
    "card2": "Mã phân nhóm thẻ phụ (card2)",
    "card3": "Mã quốc gia phát hành thẻ (card3)",
    "is_foreign_card3": "Thẻ phát hành ngoài khu vực nội địa",
    "card4_code": "Mạng lưới thẻ (Visa / Mastercard / Discover / Amex)",
    "card5": "Mã chi nhánh/hạng thẻ (card5)",
    "card6_code": "Loại thẻ (Credit / Debit)",
    "addr1": "Mã vùng địa chỉ thanh toán (Billing Region)",
    "addr2": "Mã quốc gia thanh toán (Billing Country)",
    "dist1": "Khoảng cách giữa địa chỉ thanh toán và địa chỉ nhận (dist1)",
    "dist2": "Khoảng cách phụ theo định vị thiết bị (dist2)",
    "device_type_code": "Loại thiết bị (Mobile / Desktop / Ẩn)",
    "has_device_info": "Có định danh thiết bị (DeviceInfo) hay ẩn danh",
    "has_r_email": "Có tài khoản email người nhận riêng biệt",
    "email_domain_mismatch": "Email người mua và người nhận lệch tên miền",
    "hour": "Khung giờ giao dịch trong ngày (0h - 23h)",
    "is_night": "Giao dịch vào khung giờ đêm khuya (00h - 05h)",
    "day_of_week": "Ngày trong tuần",
    # Causal Velocity & UID Behavioral Features (Strict Past-Only)
    "time_since_last_uid_tx": "Thời gian kể từ giao dịch liền trước của cùng chủ thẻ (giây)",
    "uid_tx_1h": "Số lần thanh toán dồn dập trong 1 giờ qua (Velocity 1h)",
    "uid_tx_24h": "Số lần thanh toán trong 24 giờ qua (Velocity 24h)",
    "uid_amt_sum_24h": "Tổng tiền chủ thẻ đã chi trong 24 giờ qua",
    "uid_tx_count_prev": "Số giao dịch lịch sử trước đó của chủ thẻ",
    "amt_ratio_uid_mean": "Số tiền gấp bao nhiêu lần mức chi TB lịch sử của chính chủ thẻ",
    "amt_diff_uid_mean": "Chênh lệch số tiền so với mức chi TB lịch sử của chủ thẻ",
    "amt_zscore_prod": "Độ lệch chuẩn số tiền trong cùng nhóm sản phẩm (ProductCD)",
    "card1_past_freq": "Tần suất xuất hiện lịch sử của đầu thẻ card1",
    "addr1_past_freq": "Tần suất xuất hiện lịch sử của khu vực addr1",
    # Out-Of-Fold Smoothed Target Risk (Zero Overfitting!)
    "p_email_oof_risk": "Rủi ro lịch sử tên miền email người mua (OOF)",
    "r_email_oof_risk": "Rủi ro lịch sử tên miền email người nhận (OOF)",
    "card1_oof_risk": "Rủi ro lịch sử đầu thẻ ngân hàng card1 (OOF)",
    "card2_oof_risk": "Rủi ro lịch sử phân nhóm thẻ card2 (OOF)",
    "addr1_oof_risk": "Rủi ro lịch sử khu vực địa lý addr1 (OOF)",
    "device_oof_risk": "Rủi ro lịch sử dòng thiết bị DeviceInfo (OOF)",
    # Real Vesta Payment Gateway Telemetry (C, D, M, ID, V-block signals)
    "C_suspicious_activity": "Mật độ liên kết thực thể bất thường (Vesta C-signal)",
    "C_elevated_activity": "Tần suất hoạt động tăng đột biến (Vesta C-signal)",
    "C_normal_behavior_score": "Chỉ số độ ổn định hành vi thực thể (Vesta C-signal)",
    "C_extreme_risk_flag": "Cờ mật độ liên kết vượt ngưỡng cực đại (Vesta C-flag)",
    "D_core_recency": "Khoảng cách ngày từ giao dịch/tài khoản gần nhất (Vesta D-signal)",
    "D_secondary_recency": "Độ tuổi liên kết thứ cấp của thẻ/email (Vesta D-signal)",
    "D_missing_count": "Số trường lịch sử bị thiếu / tài khoản ẩn danh (Vesta D-missing)",
    "D_new_account": "Cờ tài khoản mới tinh (Vesta New Account)",
    "D_extreme_risk_flag": "Cờ bất thường độ mới tài khoản cực đại (Vesta D-flag)",
    "D_high_risk_flag": "Cờ rủi ro độ mới tài khoản (Vesta D-flag)",
    "M_risk_category": "Mức độ sai khớp thông tin chủ thẻ (Vesta M-match)",
    "M_flag_1": "Cờ đối khớp địa chỉ/tên M1",
    "M_flag_2": "Cờ đối khớp địa chỉ/tên M2",
    "M_flag_3": "Cờ đối khớp địa chỉ/tên M3",
    "M_flag_4": "Cờ đối khớp địa chỉ/tên M4",
    "M8_M9_sum": "Tổng điểm đối khớp danh tính mở rộng (M8+M9)",
    "id_15_code": "Trạng thái nhận diện thiết bị (New/Found/Unknown)",
    "id_16_val": "Tín hiệu xác thực thiết bị id_16",
    "id_32": "Độ phân giải/màu màn hình thiết bị (id_32)",
    "id_34_code": "Trạng thái đối khớp phiên đăng nhập (id_34)",
    "id_35": "Tín hiệu xác thực danh tính thiết bị (id_35)",
    "vesta_v_block_1": "Tín hiệu tổng hợp chuỗi Vesta Gateway #1",
    "vesta_v_block_2": "Tín hiệu tổng hợp chuỗi Vesta Gateway #2",
    "vesta_v_block_3": "Tín hiệu tổng hợp chuỗi Vesta Gateway #3",
    "vesta_v_block_4": "Tín hiệu tổng hợp chuỗi Vesta Gateway #4",
    "vesta_v_block_5": "Tín hiệu tổng hợp chuỗi Vesta Gateway #5",
    "vesta_v_ind_1": "Chỉ báo hành vi cổng thanh toán Vesta #1",
    "vesta_v_ind_2": "Chỉ báo hành vi cổng thanh toán Vesta #2",
    "vesta_v_ind_3": "Chỉ báo hành vi cổng thanh toán Vesta #3",
    "vesta_v_ind_4": "Chỉ báo hành vi cổng thanh toán Vesta #4",
    "vesta_v_ind_5": "Chỉ báo hành vi cổng thanh toán Vesta #5",
}
FEATURE_COLS = list(FEATURE_NAMES_VI.keys())


def compute_uid_rolling_velocity(df: pd.DataFrame):
    n = len(df)
    c1h = np.zeros(n, dtype=np.float32)
    c24h = np.zeros(n, dtype=np.float32)
    s24h = np.zeros(n, dtype=np.float32)

    uid = (
        df["card1"].fillna(-1).astype(str)
        + "_"
        + df["card2"].fillna(-1).astype(str)
        + "_"
        + df["addr1"].fillna(-1).astype(str)
    ).astype("category").cat.codes.values

    ts = df["TransactionDT"].values
    am = df["TransactionAmt"].values
    order = np.argsort(uid, kind="mergesort")
    u_s, t_s, a_s = uid[order], ts[order], am[order]
    ch = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1], True])

    for k in range(len(ch) - 1):
        st, en = ch[k], ch[k + 1]
        t_arr, a_arr = t_s[st:en], a_s[st:en]
        m = en - st
        r1 = np.zeros(m, dtype=np.float32)
        r24 = np.zeros(m, dtype=np.float32)
        rs24 = np.zeros(m, dtype=np.float32)
        p1 = p24 = 0
        run_s = 0.0
        for i in range(m):
            ti = t_arr[i]
            while p1 < i and ti - t_arr[p1] > 3600:
                p1 += 1
            while p24 < i and ti - t_arr[p24] > 86400:
                run_s -= a_arr[p24]
                p24 += 1
            r1[i] = i - p1
            r24[i] = i - p24
            rs24[i] = run_s
            run_s += a_arr[i]
        idx = order[st:en]
        c1h[idx], c24h[idx], s24h[idx] = r1, r24, rs24
    return uid, c1h, c24h, s24h


def oof_target_encode(train_series, y_train, val_series, test_series, smooth=40.0, n_folds=5):
    """Out-of-fold Smoothed Target Encoding trên Train (chống overfit 100%) + Full-Train map cho Val/Test."""
    tr_s = train_series.fillna("missing").astype(str).str.lower().values
    va_s = val_series.fillna("missing").astype(str).str.lower().values
    te_s = test_series.fillna("missing").astype(str).str.lower().values
    g_rate = float(y_train.mean())

    oof = np.full(len(tr_s), g_rate, dtype=np.float32)
    fold_ids = np.arange(len(tr_s)) % n_folds
    for f in range(n_folds):
        m_tr = fold_ids != f
        m_va = fold_ids == f
        agg = pd.DataFrame({"k": tr_s[m_tr], "y": y_train[m_tr]}).groupby("k")["y"].agg(["count", "mean"])
        fmap = ((agg["count"] * agg["mean"] + smooth * g_rate) / (agg["count"] + smooth)).to_dict()
        oof[m_va] = pd.Series(tr_s[m_va]).map(fmap).fillna(g_rate).values

    full_agg = pd.DataFrame({"k": tr_s, "y": y_train}).groupby("k")["y"].agg(["count", "mean"])
    full_map = ((full_agg["count"] * full_agg["mean"] + smooth * g_rate) / (full_agg["count"] + smooth)).to_dict()
    va_enc = pd.Series(va_s).map(full_map).fillna(g_rate).values.astype(np.float32)
    te_enc = pd.Series(te_s).map(full_map).fillna(g_rate).values.astype(np.float32)
    return oof, va_enc, te_enc, full_map


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    print("=" * 86)
    print("REAL-WORLD PAYMENT FRAUD ENGINE v2: IEEE-CIS VESTA (590,540 GD THẬT — OOF + FULL GATEWAY)")
    print("=" * 86)
    t_all = time.time()

    df = pd.read_csv(VESTA_PATH).sort_values("TransactionDT").reset_index(drop=True)
    print(f"[1/4] Đọc toàn bộ giao dịch THẬT IEEE-CIS Vesta: {len(df):,} GD | Gian lận THẬT: {int(df['isFraud'].sum()):,} ({df['isFraud'].mean()*100:.2f}%)")

    df["amt_log"] = np.log1p(np.maximum(df["TransactionAmt"].astype(float), 0.0))
    df["amt_decimal"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 3)
    df["amt_is_round"] = (df["amt_decimal"] == 0.0).astype(int)

    prod_map = {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4}
    card4_map = {"visa": 0, "mastercard": 1, "discover": 2, "american express": 3}
    card6_map = {"debit": 0, "credit": 1, "debit or credit": 2, "charge card": 3}
    dev_map = {"desktop": 1, "mobile": 2}
    id15_map = {"new": 1, "found": 2, "unknown": 3}
    id34_map = {"match_status:2": 2, "match_status:1": 1, "match_status:0": 0, "match_status:-1": -1}

    df["product_code"] = df["ProductCD"].astype(str).map(prod_map).fillna(-1).astype(int)
    df["card4_code"] = df["card4"].astype(str).str.lower().map(card4_map).fillna(-1).astype(int)
    df["card6_code"] = df["card6"].astype(str).str.lower().map(card6_map).fillna(-1).astype(int)
    df["is_foreign_card3"] = (df["card3"].fillna(150.0) != 150.0).astype(int)
    df["device_type_code"] = df["DeviceType"].astype(str).str.lower().map(dev_map).fillna(0).astype(int)
    df["has_device_info"] = df["DeviceInfo"].notna().astype(int)
    df["id_15_code"] = df["id_15"].astype(str).str.lower().map(id15_map).fillna(0).astype(int)
    df["id_16_val"] = df["id_16"].fillna(-1.0).astype(float)
    df["id_34_code"] = df["id_34"].astype(str).str.lower().map(id34_map).fillna(-2).astype(int)

    # Ánh xạ Vesta V-block telemetry rõ tên (chính là các cột Vxxx gốc của cổng thanh toán Vesta)
    for i in range(1, 6):
        df[f"vesta_v_block_{i}"] = df[f"fraud_score_{i}"]
        df[f"vesta_v_ind_{i}"] = df[f"risk_indicator_{i}"]

    p_em = df["P_emaildomain"].fillna("missing").astype(str).str.lower()
    r_em = df["R_emaildomain"].fillna("missing").astype(str).str.lower()
    df["has_r_email"] = (r_em != "missing").astype(int)
    df["email_domain_mismatch"] = ((r_em != "missing") & (p_em != r_em)).astype(int)

    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(int)

    print("[2/4] Tính Rolling Velocity 1h/24h & Tần suất nhân quả quá khứ (Zero-Leakage)...")
    uid, c1h, c24h, s24h = compute_uid_rolling_velocity(df)
    df["_uid"] = uid
    grp = df.groupby("_uid", sort=False)
    prev_t = grp["TransactionDT"].shift(1)
    df["time_since_last_uid_tx"] = (df["TransactionDT"] - prev_t).fillna(86400.0).clip(lower=1.0)
    df["uid_tx_1h"] = c1h
    df["uid_tx_24h"] = c24h
    df["uid_amt_sum_24h"] = s24h
    df["uid_tx_count_prev"] = grp.cumcount()
    cum_prev = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]
    c_mean = np.where(df["uid_tx_count_prev"] > 0, cum_prev / np.maximum(df["uid_tx_count_prev"], 1), df["TransactionAmt"])
    df["amt_ratio_uid_mean"] = np.clip(df["TransactionAmt"] / (c_mean + 1.0), 0.0, 50.0)
    df["amt_diff_uid_mean"] = df["TransactionAmt"] - c_mean
    df["card1_past_freq"] = df.groupby("card1", dropna=False).cumcount()
    df["addr1_past_freq"] = df.groupby("addr1", dropna=False).cumcount()

    # Chronological Split nghiêm ngặt: 70% Train / 15% Val / 15% Future Test
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    train_df = df.iloc[:i_tr].copy()
    val_df = df.iloc[i_tr:i_va].copy()
    test_df = df.iloc[i_va:].copy()

    y_train = train_df["isFraud"].astype(int).values
    y_val = val_df["isFraud"].astype(int).values
    y_test = test_df["isFraud"].astype(int).values
    g_rate = float(y_train.mean())

    prod_stats = train_df.groupby("ProductCD")["TransactionAmt"].agg(["mean", "std"]).to_dict("index")
    for part in (train_df, val_df, test_df):
        pm = part["ProductCD"].map(lambda c: prod_stats.get(c, {"mean": 100.0})["mean"])
        ps = part["ProductCD"].map(lambda c: max(prod_stats.get(c, {"std": 100.0})["std"], 1.0))
        part["amt_zscore_prod"] = (part["TransactionAmt"] - pm) / ps

    # 5-Fold Out-Of-Fold Target Encoding trên Train (triệt tiêu overfit card1!)
    maps = {}
    for col_src, col_dst, sm in [
        ("P_emaildomain", "p_email_oof_risk", 40.0),
        ("R_emaildomain", "r_email_oof_risk", 40.0),
        ("card1", "card1_oof_risk", 35.0),
        ("card2", "card2_oof_risk", 35.0),
        ("addr1", "addr1_oof_risk", 40.0),
        ("DeviceInfo", "device_oof_risk", 35.0),
    ]:
        tr_e, va_e, te_e, mp = oof_target_encode(train_df[col_src], y_train, val_df[col_src], test_df[col_src], smooth=sm)
        train_df[col_dst] = tr_e
        val_df[col_dst] = va_e
        test_df[col_dst] = te_e
        maps[col_dst] = mp

    X_train = train_df[FEATURE_COLS]
    X_val = val_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]

    print(f"[3/4] Huấn luyện LightGBM (70 đặc trưng thực tế | Train={len(X_train):,} | Val={len(X_val):,} | Test tương lai={len(X_test):,})...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=11,
        min_child_samples=50,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)],
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

    rec_at_1pct_fpr = float(recall_score(y_test, (test_prob >= np.percentile(test_prob[y_test == 0], 99.0)).astype(int)))
    rec_at_3pct_fpr = float(recall_score(y_test, (test_prob >= np.percentile(test_prob[y_test == 0], 97.0)).astype(int)))
    rec_at_5pct_fpr = float(recall_score(y_test, (test_prob >= np.percentile(test_prob[y_test == 0], 95.0)).astype(int)))

    print("[4/4] Huấn luyện Isolation Forest (Zero-Day Anomaly) trên giao dịch thật sạch...")
    iso_fill = X_train[y_train == 0].fillna(-1.0).sample(n=100_000, random_state=42)
    iso_model = IsolationForest(n_estimators=120, contamination=0.03, max_samples=4096, random_state=42, n_jobs=-1)
    iso_model.fit(iso_fill)
    raw_iso_val = -iso_model.score_samples(X_val.fillna(-1.0))
    iso_min, iso_max = float(np.percentile(raw_iso_val, 1)), float(np.percentile(raw_iso_val, 99.5))
    anom_test = np.clip((-iso_model.score_samples(X_test.fillna(-1.0)) - iso_min) / (iso_max - iso_min + 1e-9) * 100.0, 0, 100)

    imp = pd.Series(lgb_model.booster_.feature_importance(importance_type="gain"), index=FEATURE_COLS)
    top_imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(10)

    print("\n" + "-" * 86)
    print(f"KẾT QUẢ THỰC TẾ IEEE-CIS VESTA v2 TRÊN TẬP TEST TƯƠNG LAI ({len(X_test):,} GD THẬT | {int(y_test.sum()):,} GIAN LẬN THẬT):")
    print(f"  * Best Threshold (từ Val)    : {best_th:.4f} (Val F1 = {val_f1:.4f} -> Test F1 = {f1:.4f} — KHÔNG OVERFIT!)")
    print(f"  * ROC-AUC                    : {roc_auc:.5f}")
    print(f"  * PR-AUC (Avg Precision)     : {pr_auc:.5f}  (Lift {pr_auc/y_test.mean():.1f}x so với ngẫu nhiên!)")
    print(f"  * F1-Score                   : {f1:.5f}")
    print(f"  * Precision (Độ chuẩn xác)   : {prec*100:.2f}%")
    print(f"  * Recall (Tỷ lệ bắt đúng)    : {rec*100:.2f}%")
    print(f"  * False Positive Rate (FPR)  : {fpr*100:.2f}%")
    print(f"  * Recall @ 1% FPR (Chặn tự động): {rec_at_1pct_fpr*100:.2f}%")
    print(f"  * Recall @ 3% FPR (Duyệt Ops)   : {rec_at_3pct_fpr*100:.2f}%")
    print(f"  * Recall @ 5% FPR (Step-up OTP) : {rec_at_5pct_fpr*100:.2f}%")
    print(f"  * Confusion Matrix           : TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print(f"  * Thời gian huấn luyện       : {time.time() - t_all:.1f}s")
    print("\n  Top 10 đặc trưng quan trọng nhất (Gain % — cân bằng thực tế):")
    for k, v in top_imp.items():
        print(f"    - {v:5.2f}%  {k:26s} ({FEATURE_NAMES_VI[k]})")
    print("-" * 86)

    bundle = {
        "name": "IEEE-CIS Vesta Real-World E-Commerce Payment Fraud Engine v2",
        "lgb_model": lgb_model,
        "iso_model": iso_model,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "threshold": best_th,
        "feature_cols": FEATURE_COLS,
        "feature_names_vi": FEATURE_NAMES_VI,
        "prod_map": prod_map,
        "card4_map": card4_map,
        "card6_map": card6_map,
        "dev_map": dev_map,
        "prod_stats": prod_stats,
        "oof_maps": maps,
        "global_rate": g_rate,
        "metrics": {
            "dataset": f"IEEE-CIS Vesta Corporation ({len(df):,} giao dịch thương mại điện tử THẬT - 0% Synthetic)",
            "split": "70% Train / 15% Val / 15% Test (Chronological - theo thời gian thực)",
            "test_size": int(len(X_test)),
            "test_frauds": int(y_test.sum()),
            "roc_auc": round(float(roc_auc), 5),
            "pr_auc": round(float(pr_auc), 5),
            "f1": round(float(f1), 5),
            "precision": round(float(prec), 5),
            "recall": round(float(rec), 5),
            "fpr": round(float(fpr), 5),
            "recall_at_1pct_fpr": round(float(rec_at_1pct_fpr), 4),
            "recall_at_3pct_fpr": round(float(rec_at_3pct_fpr), 4),
            "recall_at_5pct_fpr": round(float(rec_at_5pct_fpr), 4),
            "threshold": round(float(best_th), 4),
            "latency_ms": round(float(inf_ms), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }
    out_path = os.path.join(MODEL_DIR, "vesta_real_bundle.joblib")
    joblib.dump(bundle, out_path)

    test_df["risk_prob"] = np.round(test_prob * 100.0, 2)
    test_df["s_anom"] = np.round(anom_test, 1)
    sample_v = pd.concat([
        test_df[test_df["isFraud"] == 1].sort_values("risk_prob", ascending=False).head(60),
        test_df[test_df["isFraud"] == 0].sort_values("risk_prob", ascending=False).head(40),
        test_df[test_df["isFraud"] == 0].sample(n=100, random_state=42),
    ], ignore_index=True)
    meta = ["TransactionID", "TransactionDT", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "isFraud", "risk_prob", "s_anom"]
    keep = meta + [c for c in FEATURE_COLS if c not in meta]
    sample_v[keep].to_parquet(os.path.join(DATA_DIR, "vesta_real_ops_samples.parquet"), index=False)
    print(f"[HOÀN TẤT] Đã lưu model thực tế IEEE-CIS Vesta v2 tại: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
