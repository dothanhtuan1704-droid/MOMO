import os
import sys
import time
import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REAL_DIR = os.path.join(DATA_DIR, "real_world_test")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from train_vesta_real import compute_uid_rolling_velocity
from train_phiusiil import extract_url_batch


def verify_vesta_real():
    print("=" * 88)
    print("KIỂM THỬ SÂU (STRESS TEST) MODEL 1: IEEE-CIS VESTA REAL-WORLD PAYMENT FRAUD")
    print("=" * 88)
    b = joblib.load(os.path.join(MODEL_DIR, "vesta_real_bundle.joblib"))
    lgb_m = b["lgb_model"]
    iso_m = b["iso_model"]
    th = b["threshold"]
    fcols = b["feature_cols"]
    names_vi = b["feature_names_vi"]
    maps = b["oof_maps"]
    g_rate = b["global_rate"]
    prod_stats = b["prod_stats"]

    df = pd.read_csv(os.path.join(REAL_DIR, "vesta_real_ieee_cis.csv")).sort_values("TransactionDT").reset_index(drop=True)
    df["amt_log"] = np.log1p(np.maximum(df["TransactionAmt"].astype(float), 0.0))
    df["amt_decimal"] = np.round(df["TransactionAmt"] - np.floor(df["TransactionAmt"]), 3)
    df["amt_is_round"] = (df["amt_decimal"] == 0.0).astype(int)

    df["product_code"] = df["ProductCD"].astype(str).map(b["prod_map"]).fillna(-1).astype(int)
    df["card4_code"] = df["card4"].astype(str).str.lower().map(b["card4_map"]).fillna(-1).astype(int)
    df["card6_code"] = df["card6"].astype(str).str.lower().map(b["card6_map"]).fillna(-1).astype(int)
    df["is_foreign_card3"] = (df["card3"].fillna(150.0) != 150.0).astype(int)
    df["device_type_code"] = df["DeviceType"].astype(str).str.lower().map(b["dev_map"]).fillna(0).astype(int)
    df["has_device_info"] = df["DeviceInfo"].notna().astype(int)
    id15_map = {"new": 1, "found": 2, "unknown": 3}
    id34_map = {"match_status:2": 2, "match_status:1": 1, "match_status:0": 0, "match_status:-1": -1}
    df["id_15_code"] = df["id_15"].astype(str).str.lower().map(id15_map).fillna(0).astype(int)
    df["id_16_val"] = df["id_16"].fillna(-1.0).astype(float)
    df["id_34_code"] = df["id_34"].astype(str).str.lower().map(id34_map).fillna(-2).astype(int)

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

    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    train_uids = set(df.iloc[:i_tr]["_uid"].tolist())
    train_card1s = set(df.iloc[:i_tr]["card1"].dropna().tolist())
    test_df = df.iloc[i_va:].copy().reset_index(drop=True)

    pm = test_df["ProductCD"].map(lambda c: prod_stats.get(c, {"mean": 100.0})["mean"])
    ps = test_df["ProductCD"].map(lambda c: max(prod_stats.get(c, {"std": 100.0})["std"], 1.0))
    test_df["amt_zscore_prod"] = (test_df["TransactionAmt"] - pm) / ps
    for col_src, col_dst in [
        ("P_emaildomain", "p_email_oof_risk"),
        ("R_emaildomain", "r_email_oof_risk"),
        ("card1", "card1_oof_risk"),
        ("card2", "card2_oof_risk"),
        ("addr1", "addr1_oof_risk"),
        ("DeviceInfo", "device_oof_risk"),
    ]:
        test_df[col_dst] = test_df[col_src].fillna("missing").astype(str).str.lower().map(maps[col_dst]).fillna(g_rate)

    X_test = test_df[fcols]
    y_test = test_df["isFraud"].astype(int).values
    probs = lgb_m.predict_proba(X_test)[:, 1]
    preds = (probs >= th).astype(int)

    # 1A: Hiệu quả cứu tiền thực tế ($USD) trên 88,581 giao dịch tương lai
    total_fraud_usd = test_df.loc[y_test == 1, "TransactionAmt"].sum()
    saved_usd_default = test_df.loc[(y_test == 1) & (preds == 1), "TransactionAmt"].sum()
    th_1fpr = np.percentile(probs[y_test == 0], 99.0)
    th_5fpr = np.percentile(probs[y_test == 0], 95.0)
    saved_usd_1fpr = test_df.loc[(y_test == 1) & (probs >= th_1fpr), "TransactionAmt"].sum()
    saved_usd_5fpr = test_df.loc[(y_test == 1) & (probs >= th_5fpr), "TransactionAmt"].sum()

    print(f"\n[TEST 1A] TÁC ĐỘNG TÀI CHÍNH THỰC TẾ TRÊN 88,581 GIAO DỊCH TƯƠNG LAI:")
    print(f"  * Tổng tiền gian lận thực tế trong tập Test : ${total_fraud_usd:,.2f}")
    print(f"  * Cứu được ở ngưỡng tối ưu F1 (FPR 2.02%)   : ${saved_usd_default:,.2f} ({saved_usd_default/total_fraud_usd*100:.1f}% tổng tiền thiệt hại)")
    print(f"  * Cứu được ở chế độ Chặn tự động (1% FPR)   : ${saved_usd_1fpr:,.2f} ({saved_usd_1fpr/total_fraud_usd*100:.1f}% tổng tiền thiệt hại)")
    print(f"  * Cứu được ở chế độ Step-up OTP (5% FPR)    : ${saved_usd_5fpr:,.2f} ({saved_usd_5fpr/total_fraud_usd*100:.1f}% tổng tiền thiệt hại)")

    # 1B: Kiểm thử trên CHỦ THẺ HOÀN TOÀN MỚI (Unseen UID & Unseen card1 — chưa bao giờ xuất hiện trong Train!)
    unseen_uid_mask = ~test_df["_uid"].isin(train_uids)
    unseen_c1_mask = ~test_df["card1"].isin(train_card1s)

    for label, mask in [
        ("Chủ thẻ hoàn toàn mới (Unseen UID)", unseen_uid_mask),
        ("Đầu thẻ ngân hàng hoàn toàn mới (Unseen card1)", unseen_c1_mask),
    ]:
        yt = y_test[mask]
        pt = probs[mask]
        yp = (pt >= th).astype(int)
        print(f"\n[TEST 1B] COLD-START — {label}:")
        print(
            f"  * Samples={len(yt):,} | Frauds={int(yt.sum()):,} ({yt.mean()*100:.2f}%) | "
            f"ROC-AUC={roc_auc_score(yt, pt):.4f} | PR-AUC={average_precision_score(yt, pt):.4f} | "
            f"F1={f1_score(yt, yp):.4f} | Prec={precision_score(yt, yp)*100:.1f}% | Rec={recall_score(yt, yp)*100:.1f}%"
        )

    # 1C: Kiểm thử theo loại thiết bị (Mobile vs Desktop) & nhóm sản phẩm (ProductCD)
    print("\n[TEST 1C] PHÂN TÍCH ĐỘ ỔN ĐỊNH THEO THIẾT BỊ & LOẠI SẢN PHẨM:")
    for dev in ["mobile", "desktop"]:
        m = test_df["DeviceType"].astype(str).str.lower() == dev
        yt, pt = y_test[m], probs[m]
        print(f"  * Thiết bị {dev.upper():8s}: Samples={len(yt):6,} | Frauds={int(yt.sum()):4,} ({yt.mean()*100:4.1f}%) | ROC-AUC={roc_auc_score(yt, pt):.4f} | PR-AUC={average_precision_score(yt, pt):.4f}")
    for prod in ["W", "C", "R", "H", "S"]:
        m = test_df["ProductCD"] == prod
        yt, pt = y_test[m], probs[m]
        if yt.sum() > 5:
            print(f"  * Sản phẩm {prod:8s}: Samples={len(yt):6,} | Frauds={int(yt.sum()):4,} ({yt.mean()*100:4.1f}%) | ROC-AUC={roc_auc_score(yt, pt):.4f} | PR-AUC={average_precision_score(yt, pt):.4f}")

    # 1D: Demo chấm điểm thời gian thực + Giải thích SHAP tiếng Việt trên 1 giao dịch gian lận thật
    explainer = shap.TreeExplainer(lgb_m)
    fraud_caught_idx = np.where((y_test == 1) & (probs >= 0.80))[0][0]
    row_x = X_test.iloc[[fraud_caught_idx]]
    t0 = time.time()
    p_one = float(lgb_m.predict_proba(row_x)[0, 1]) * 100.0
    sv = explainer.shap_values(row_x)
    sv_row = sv[1][0] if isinstance(sv, list) else sv[0]
    lat_ms = (time.time() - t0) * 1000.0

    pairs = sorted([(f, float(s), float(v)) for f, s, v in zip(fcols, sv_row, row_x.iloc[0].values) if s > 0], key=lambda z: z[1], reverse=True)
    tot = sum(z[1] for z in pairs) + 1e-9
    meta_r = test_df.iloc[fraud_caught_idx]
    print(f"\n[TEST 1D] LIVE INFERENCE + SHAP TIẾNG VIỆT (1 GIAO DỊCH GIAN LẬN THẬT — Độ trễ: {lat_ms:.1f} ms):")
    print(f"  * TxID={meta_r['TransactionID']} | Số tiền=${meta_r['TransactionAmt']:,.2f} | Thẻ={meta_r['card4']}/{meta_r['card6']} | Email={meta_r['P_emaildomain']} | Thiết bị={meta_r['DeviceType']} ({meta_r['DeviceInfo']})")
    print(f"  * Xác suất gian lận AI: {p_one:.1f}% -> KHUYẾN NGHỊ: CHẶN & ĐẨY BÀN FRAUD OPS")
    for f, s, v in pairs[:4]:
        print(f"     + {names_vi[f]} (giá trị={v:.2f}): đóng góp +{s/tot*100:.0f}% rủi ro")


def verify_url_real():
    print("\n" + "=" * 88)
    print("KIỂM THỬ SÂU (STRESS TEST) MODEL 2: REAL-WORLD LIVE ANTI-PHISHING URL ENGINE")
    print("=" * 88)
    b = joblib.load(os.path.join(MODEL_DIR, "phiusiil_bundle.joblib"))
    th = b["threshold"]
    explainer = shap.TreeExplainer(b["lgb_model"])

    def score_batch(urls):
        X_lex, norms, _, tlds = extract_url_batch(urls)
        X_lex["tld_phish_rate"] = [b["tld_rate_map"].get(t, b["global_rate"]) for t in tlds]
        X_lex["tfidf_char_prob"] = b["char_clf"].predict_proba(b["char_vec"].transform(norms))[:, 1]
        X_lex["tfidf_word_prob"] = b["word_clf"].predict_proba(b["word_vec"].transform(norms))[:, 1]
        X_final = X_lex[b["feature_cols"]]
        probs = b["lgb_model"].predict_proba(X_final)[:, 1]
        return probs, X_final

    # 2A: Hard Adversarial OpenPhish Live (CHỈ lấy các link lừa đảo tinh vi dùng HTTPS + đuôi uy tín .com/.org/.net!)
    with open(os.path.join(REAL_DIR, "openphish_live.txt"), "r", encoding="utf-8", errors="ignore") as f:
        op_all = [x.strip() for x in f if x.strip().startswith("http")]
    op_hard = [
        u for u in op_all
        if u.lower().startswith("https://")
        and any(u.lower().split("/")[2].endswith(t) for t in (".com", ".org", ".net"))
    ]
    p_op_all, _ = score_batch(op_all)
    p_op_hard, _ = score_batch(op_hard)
    print(f"\n[TEST 2A] LIVE OPENPHISH HÔM NAY — BAO GỒM CẢ LINK LỪA ĐẢO TINH VI (HTTPS + .com/.org/.net):")
    print(f"  * Toàn bộ OpenPhish Live hôm nay ({len(op_all)} link)           : Bắt trúng {(p_op_all>=th).mean()*100:.2f}% ({int((p_op_all>=th).sum())}/{len(op_all)})")
    print(f"  * Riêng nhóm TINH VI (HTTPS + .com/.org/.net — {len(op_hard)} link) : Bắt trúng {(p_op_hard>=th).mean()*100:.2f}% ({int((p_op_hard>=th).sum())}/{len(op_hard)})")
    print("  * Ví dụ 3 link lừa đảo sống tinh vi vừa bắt được hôm nay:")
    caught_hard = [(u, p) for u, p in zip(op_hard, p_op_hard) if p >= th]
    for u, p in caught_hard[:3]:
        print(f"     - [{p*100:5.1f}% PHISHING] {u}")

    # 2B: Bộ 10 kịch bản lừa đảo thực tế tại Việt Nam (Fintech/Ngân hàng/Cổng dịch vụ công) vs 10 trang thật VN
    vn_cases = [
        ("SAFE", "https://momo.vn"),
        ("SAFE", "https://vietcombank.com.vn/ca-nhan"),
        ("SAFE", "https://dichvucong.gov.vn"),
        ("SAFE", "https://bidv.com.vn/vn/ca-nhan"),
        ("SAFE", "https://techcombank.com/khach-hang-ca-nhan"),
        ("SAFE", "https://zalopay.vn"),
        ("SAFE", "https://vnpay.vn"),
        ("SAFE", "https://mbbank.com.vn"),
        ("SAFE", "https://vnexpress.net/kinh-doanh"),
        ("SAFE", "https://tiki.vn"),
        ("PHISH", "https://momo-trian-khachhang2026.xyz/nhan-qua"),
        ("PHISH", "http://vietcombank-digibank-verify.top/login/otp"),
        ("PHISH", "https://dichvucong-vneid-capnhat.online/xacminh.php"),
        ("PHISH", "https://bidv-smartbanking-secure.firebaseapp.com/signin"),
        ("PHISH", "http://103.45.88.19/techcombank/verify-card.html"),
        ("PHISH", "https://zalopay-hoantien-99k.vip/claim"),
        ("PHISH", "https://mb-bank-khoiphuc-taikhoan.club/dangnhap"),
        ("PHISH", "https://vnpay.com.xacminh-giaodich.icu/confirm"),
        ("PHISH", "http://giao-hang-tiet-kiem-boithuong.site/banking/login"),
        ("PHISH", "https://baohiemxahoi-vssid-update.cc/download/app.apk"),
    ]
    t0 = time.time()
    p_vn, X_vn = score_batch([u for _, u in vn_cases])
    avg_ms = (time.time() - t0) / len(vn_cases) * 1000.0
    ok_cnt = 0
    print(f"\n[TEST 2B] BỘ 20 KỊCH BẢN THỰC TẾ VIỆT NAM (10 TRANG THẬT vs 10 LINK LỪA ĐẢO THỰC TẾ — {avg_ms:.2f} ms/URL):")
    for (truth, u), p in zip(vn_cases, p_vn):
        pred = "PHISH" if p >= th else "SAFE"
        mark = "OK " if pred == truth else "SAI"
        ok_cnt += int(pred == truth)
        print(f"  [{mark}] Thật={truth:5s} | AI={pred:5s} ({p*100:5.1f}%) | {u}")
    print(f"  -> ĐỘ CHÍNH XÁC TỔNG HỢP TRÊN 20 CA VIỆT NAM (THUẦN ML): {ok_cnt}/{len(vn_cases)} ({ok_cnt/len(vn_cases)*100:.1f}%)")

    # 2C: SHAP giải thích 1 link lừa đảo VNeID tiếng Việt
    idx_demo = 12  # dichvucong-vneid-capnhat.online/xacminh.php
    sv = explainer.shap_values(X_vn.iloc[[idx_demo]])
    sv_row = sv[1][0] if isinstance(sv, list) else sv[0]
    pairs = sorted([(f, float(s), float(v)) for f, s, v in zip(b["feature_cols"], sv_row, X_vn.iloc[idx_demo].values) if s > 0], key=lambda z: z[1], reverse=True)
    tot = sum(z[1] for z in pairs) + 1e-9
    print(f"\n[TEST 2C] GIẢI THÍCH SHAP TỰ ĐỘNG CHO LINK: {vn_cases[idx_demo][1]} ({p_vn[idx_demo]*100:.1f}% LỪA ĐẢO):")
    for f, s, v in pairs[:4]:
        print(f"     + {b['feature_names_vi'][f]} (giá trị={v:.2f}): +{s/tot*100:.0f}% rủi ro")


if __name__ == "__main__":
    verify_vesta_real()
    verify_url_real()

