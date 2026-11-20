import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd
import joblib
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
EXT_DIR = os.path.join(DATA_DIR, "external_test")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(EXT_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from train_phiusiil import extract_url_batch

EXTERNAL_URLS = {
    # External Test 1: Unseen partition (0001.parquet) of full 6.36M PaySim (late-month steps)
    "paysim_ext_shard": (
        "https://huggingface.co/datasets/theman10/paysim/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet",
        os.path.join(EXT_DIR, "theman10_paysim_0001.parquet"),
    ),
    # External Test 2: COMPLETELY DIFFERENT 2022 Enterprise Bank & Card Dataset (Abdulmajeed)
    "abdulmajeed_bank": (
        "https://huggingface.co/datasets/Abdulmajeedyahya/1M-Fraud-Labeled-Synthetic-Bank-Transactions/"
        "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        os.path.join(EXT_DIR, "abdulmajeed_bank_2022.parquet"),
    ),
    # External Test 3A: COMPLETELY INDEPENDENT Phishing URL Benchmark — kmack/Phishing_urls (test split)
    "kmack_url_test": (
        "https://huggingface.co/datasets/kmack/Phishing_urls/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        os.path.join(EXT_DIR, "kmack_phishing_test.parquet"),
    ),
    # External Test 3B: COMPLETELY INDEPENDENT Phishing URL Benchmark — KaushiGihan/phishing-site-url (test split)
    "kaushi_url_test": (
        "https://huggingface.co/datasets/KaushiGihan/phishing-site-url/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        os.path.join(EXT_DIR, "kaushi_phishing_test.parquet"),
    ),
    # External Test 3C: CyRadar Vietnam Phishing Detection Dataset (shard 0, ~155MB)
    "cyradar_vn_url": (
        "https://huggingface.co/datasets/huynq3Cyradar/Phishing_Detection_Dataset/"
        "resolve/refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet",
        os.path.join(EXT_DIR, "cyradar_vn_0000.parquet"),
    ),
}


def download_all():
    print("[0/4] Đang tải các bộ dữ liệu kiểm thử ĐỘC LẬP BÊN NGOÀI (External OOD Test Sets)...")
    for key, (url, path) in EXTERNAL_URLS.items():
        if os.path.exists(path) and os.path.getsize(path) > 50_000:
            print(f"  [OK] Đã có sẵn {key}: {os.path.basename(path)} ({os.path.getsize(path)/(1024*1024):.1f} MB)")
            continue
        t0 = time.time()
        print(f"  -> Đang tải {key} ...")
        urllib.request.urlretrieve(url, path)
        print(f"     Xong ({os.path.getsize(path)/(1024*1024):.2f} MB) trong {time.time()-t0:.1f}s")


def eval_report(title, y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / max(fp + tn, 1)
    print(f"\n>>> {title}")
    print(f"    Samples = {len(y_true):,} | Positives (Fraud/Phish) = {int(y_true.sum()):,} ({y_true.mean()*100:.2f}%)")
    print(f"    ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f} | F1 = {f1:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}% | FPR = {fpr*100:.2f}%")
    print(f"    Confusion Matrix: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,} (Threshold={threshold:.4f})")
    return {"roc_auc": roc, "pr_auc": pr, "f1": f1, "prec": prec, "rec": rec, "fpr": fpr}


def test_paysim_external():
    print("\n" + "=" * 82)
    print("EXTERNAL OOD TEST 1: MODULE 1 (PAYSIM CLEAN PRE-TX MODEL)")
    print("=" * 82)
    b = joblib.load(os.path.join(MODEL_DIR, "paysim_bundle.joblib"))
    lgb_m = b["lgb_model"]
    th = b["threshold"]
    fcols = b["feature_cols"]
    type_map = b["type_map"]
    type_medians = b["type_medians"]

    # 1A: Unseen late-month partition from full 6.36M PaySim (theman10/paysim 0001.parquet)
    # Lọc bỏ bất kỳ nameOrig nào từng xuất hiện trong tập paysim_raw.parquet của mình!
    train_origs = set(pd.read_parquet(os.path.join(DATA_DIR, "paysim_raw.parquet"), columns=["nameOrig"])["nameOrig"])
    ext1 = pd.read_parquet(EXTERNAL_URLS["paysim_ext_shard"][1])
    ext1 = ext1[~ext1["nameOrig"].isin(train_origs)].sort_values("step").reset_index(drop=True)
    # Lấy tối đa 500k dòng (giữ 100% các ca fraud) để đánh giá nhanh & toàn diện
    if len(ext1) > 500_000:
        f_part = ext1[ext1["isFraud"] == 1]
        n_part = ext1[ext1["isFraud"] == 0].sample(n=500_000 - len(f_part), random_state=42)
        ext1 = pd.concat([f_part, n_part], ignore_index=True).sort_values("step").reset_index(drop=True)

    df = ext1.copy()
    df["type_code"] = df["type"].map(type_map).fillna(0).astype(int)
    df["is_risky_type"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    df["hour_of_day"] = (df["step"] % 24).astype(int)
    df["is_night_tx"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["amount_log"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["oldbal_org_log"] = np.log1p(np.maximum(df["oldbalanceOrg"], 0.0))
    df["oldbal_dest_log"] = np.log1p(np.maximum(df["oldbalanceDest"], 0.0))
    df["amt_to_bal_ratio"] = np.clip(np.round(df["amount"] / (df["oldbalanceOrg"] + 1.0), 2), 0.0, 5.0)
    df["projected_leftover_ratio"] = np.clip(
        np.round((df["oldbalanceOrg"] - df["amount"]) / (df["oldbalanceOrg"] + 1.0), 2), -1.0, 1.0
    )
    df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    df["amount_vs_type_median"] = df["amount"] / (df["type"].map(type_medians).fillna(1.0) + 1.0)
    df["dest_past_in_cnt"] = df.groupby("nameDest").cumcount().astype(int)
    df["orig_past_out_cnt"] = df.groupby("nameOrig").cumcount().astype(int)
    df["dest_pagerank"] = 0.0  # hoàn toàn unseen graph

    probs1a = lgb_m.predict_proba(df[fcols])[:, 1]
    eval_report("1A. PaySim External Late-Month Partition (theman10/paysim 0001 — 100% tài khoản mới)", df["isFraud"], probs1a, th)

    # 1B: COMPLETELY DIFFERENT 2022 Bank Dataset (Abdulmajeed — khác hoàn toàn tác giả, năm 2022, schema khác!)
    ab = pd.read_parquet(EXTERNAL_URLS["abdulmajeed_bank"][1])
    print(f"    [Info] Abdulmajeed 2022 shape={ab.shape}, fraud_label={ab['fraud_label'].value_counts().to_dict()}")
    rail_to_type = {"WIRE": "TRANSFER", "ACH": "TRANSFER", "ATM": "CASH_OUT", "DEBIT_CARD": "DEBIT", "CREDIT_CARD": "PAYMENT"}
    m = pd.DataFrame()
    mapped_type = ab["payment_rail"].map(rail_to_type).fillna("TRANSFER")
    m["type_code"] = mapped_type.map(type_map).fillna(4).astype(int)
    m["is_risky_type"] = mapped_type.isin(["TRANSFER", "CASH_OUT"]).astype(int)
    m["amount"] = ab["amount"].astype(float)
    m["amount_log"] = np.log1p(np.maximum(m["amount"], 0.0))
    m["oldbalanceOrg"] = np.maximum(ab["old_balance"].astype(float), 0.0)
    m["oldbal_org_log"] = np.log1p(m["oldbalanceOrg"])
    m["oldbalanceDest"] = 0.0
    m["oldbal_dest_log"] = 0.0
    m["amt_to_bal_ratio"] = np.clip(np.round(m["amount"] / (m["oldbalanceOrg"] + 1.0), 2), 0.0, 5.0)
    m["projected_leftover_ratio"] = np.clip(
        np.round((m["oldbalanceOrg"] - m["amount"]) / (m["oldbalanceOrg"] + 1.0), 2), -1.0, 1.0
    )
    m["dest_zero_before"] = 1
    m["dest_is_merchant"] = ab["payment_rail"].isin(["CREDIT_CARD", "DEBIT_CARD"]).astype(int)
    m["hour_of_day"] = ab["transaction_hour"].astype(int)
    m["is_night_tx"] = m["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    m["amount_vs_type_median"] = m["amount"] / (mapped_type.map(type_medians).fillna(10000.0) + 1.0)
    m["dest_past_in_cnt"] = ab.groupby("merchant_id").cumcount().astype(int)
    m["orig_past_out_cnt"] = ab.groupby("customer_id").cumcount().astype(int)
    m["dest_pagerank"] = 0.0

    probs1b = lgb_m.predict_proba(m[fcols])[:, 1]
    eval_report("1B. CROSS-DOMAIN ZERO-SHOT: PaySim Model -> Abdulmajeed 2022 Enterprise Bank Dataset (KHÁC HOÀN TOÀN BỘ DATA!)", ab["fraud_label"], probs1b, th)


def test_sparkov_external():
    print("\n" + "=" * 82)
    print("EXTERNAL OOD TEST 2: MODULE 2 (SPARKOV CREDIT CARD PAYMENT MODEL)")
    print("=" * 82)
    b = joblib.load(os.path.join(MODEL_DIR, "sparkov_bundle.joblib"))
    lgb_m = b["lgb_model"]
    th = b["threshold"]
    fcols = b["feature_cols"]
    cat_map = b["cat_map"]
    cat_stats = b["cat_stats"]
    cat_fraud_rate = b["cat_fraud_rate"]
    global_rate = b["global_rate"]

    # Map COMPLETELY DIFFERENT 2022 Bank/Card Dataset (Abdulmajeed) -> Sparkov feature space!
    ab = pd.read_parquet(EXTERNAL_URLS["abdulmajeed_bank"][1])
    ab["ts"] = pd.to_datetime(ab["transaction_timestamp"]).astype("int64") // 10**9
    ab = ab.sort_values("ts").reset_index(drop=True)

    mcc_to_sparkov = {
        "ONLINE_RETAIL": "shopping_net",
        "GROCERY": "grocery_pos",
        "TRAVEL": "travel",
        "RESTAURANT": "food_dining",
        "ENTERTAINMENT": "entertainment",
        "HEALTHCARE": "health_fitness",
        "UTILITIES": "home",
        "GAS_STATION": "gas_transport",
        "RETAIL": "shopping_pos",
        "LUXURY": "misc_net",
        "CRYPTO": "misc_net",
        "GAMBLING": "misc_net",
    }
    mapped_cat = ab["mcc_category"].astype(str).str.upper().map(mcc_to_sparkov).fillna("misc_net")

    x = pd.DataFrame()
    x["amt"] = ab["amount"].astype(float)
    x["amt_log"] = np.log1p(np.maximum(x["amt"], 0.0))
    x["category_code"] = mapped_cat.map(cat_map).fillna(cat_map.get("misc_net", 0)).astype(int)
    x["hour"] = ab["transaction_hour"].astype(int)
    x["hour_sin"] = np.sin(2.0 * np.pi * x["hour"] / 24.0)
    x["hour_cos"] = np.cos(2.0 * np.pi * x["hour"] / 24.0)
    x["is_night"] = x["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    x["day_of_week"] = ab["transaction_dow"].astype(int)
    x["is_weekend"] = ab["is_weekend"].astype(int)
    x["cardholder_age"] = 40.0
    x["gender_code"] = 0
    x["city_pop_log"] = np.log1p(500_000.0)
    x["distance_km"] = ab["distance_from_home_km"].astype(float).clip(lower=0.0)

    grp = ab.groupby("customer_id", sort=False)
    prev_ts = grp["ts"].shift(1)
    prev2_ts = grp["ts"].shift(2)
    prev_amt = grp["amount"].shift(1)
    prev_cat = mapped_cat.groupby(ab["customer_id"], sort=False).shift(1)

    x["time_since_last_tx_sec"] = (ab["ts"] - prev_ts).fillna(86400.0).clip(lower=1.0)
    x["time_since_last_2tx_sec"] = (ab["ts"] - prev2_ts).fillna(172800.0).clip(lower=1.0)
    x["dist_from_last_merch_km"] = x["distance_km"]
    x["travel_speed_kmh"] = np.clip(x["distance_km"] / (x["time_since_last_tx_sec"] / 3600.0 + 1e-4), 0.0, 5000.0)
    x["tx_count_last_1h"] = np.maximum(ab["transaction_velocity_1h"].astype(float) - 1.0, 0.0)
    x["tx_count_last_24h"] = x["tx_count_last_1h"]
    x["amt_sum_last_24h"] = x["tx_count_last_1h"] * prev_amt.fillna(0.0)
    x["amt_vs_prev_ratio"] = np.clip(x["amt"] / (prev_amt.fillna(x["amt"]) + 1.0), 0.0, 100.0)
    x["same_cat_as_prev"] = (mapped_cat == prev_cat).astype(int)

    x["user_tx_count_prev"] = grp.cumcount()
    cum_prev = grp["amount"].cumsum() - ab["amount"]
    u_mean = np.where(x["user_tx_count_prev"] > 0, cum_prev / np.maximum(x["user_tx_count_prev"], 1), x["amt"])
    x["amt_ratio_user_mean"] = np.clip(x["amt"] / (u_mean + 1.0), 0.0, 100.0)
    x["amt_diff_user_mean"] = x["amt"] - u_mean
    x["amt_ratio_user_cat_mean"] = x["amt_ratio_user_mean"]

    c_mean = mapped_cat.map(lambda c: cat_stats.get(c, {"mean": 70.0})["mean"])
    c_std = mapped_cat.map(lambda c: max(cat_stats.get(c, {"std": 50.0})["std"], 1.0))
    c_med = mapped_cat.map(lambda c: cat_stats.get(c, {"median": 50.0})["median"])
    x["amt_zscore_cat"] = (x["amt"] - c_mean) / c_std
    x["amt_ratio_cat_median"] = x["amt"] / (c_med + 1.0)
    x["cat_fraud_rate_train"] = mapped_cat.map(cat_fraud_rate).fillna(global_rate)
    x["merch_fraud_rate_train"] = global_rate

    probs2 = lgb_m.predict_proba(x[fcols])[:, 1]
    eval_report("2A. CROSS-DOMAIN ZERO-SHOT: Sparkov Model -> Abdulmajeed 2022 Bank/Card Dataset (KHÁC HOÀN TOÀN BỘ DATA!)", ab["fraud_label"], probs2, th)
    # Chỉ xét nhóm thanh toán thẻ trong Abdulmajeed
    card_mask = ab["payment_rail"].isin(["CREDIT_CARD", "DEBIT_CARD"])
    if card_mask.sum() > 50 and ab.loc[card_mask, "fraud_label"].sum() > 5:
        eval_report("2B. CROSS-DOMAIN ZERO-SHOT (Riêng Credit/Debit Card trong Abdulmajeed 2022)", ab.loc[card_mask, "fraud_label"], probs2[card_mask], th)


def test_url_external():
    print("\n" + "=" * 82)
    print("EXTERNAL OOD TEST 3: MODULE 3 (URL PHISHING ENGINE) TRÊN 3 BỘ DỮ LIỆU ĐỘC LẬP")
    print("=" * 82)
    b = joblib.load(os.path.join(MODEL_DIR, "phiusiil_bundle.joblib"))
    char_vec, char_clf = b["char_vec"], b["char_clf"]
    word_vec, word_clf = b["word_vec"], b["word_clf"]
    lgb_m = b["lgb_model"]
    tld_map = b["tld_rate_map"]
    g_rate = b["global_rate"]
    th = b["threshold"]
    fcols = b["feature_cols"]

    # Tải tập các URL đã có trong Train (PhiUSIIL + Mitake + Pirocheto) để LOẠI BỎ bất kỳ URL trùng nào!
    seen_urls = set()
    for p_name, col in [("phiusiil_raw.parquet", "phiusiil_url"), ("mitake_urls.parquet", "url"), ("pirocheto_urls.parquet", "url")]:
        fp = os.path.join(DATA_DIR, p_name)
        if os.path.exists(fp):
            s = pd.read_parquet(fp, columns=[col])[col].astype(str).str.lower().str.replace(r"^https?://(www\.)?", "", regex=True).str.rstrip("/")
            seen_urls.update(s.tolist())
    print(f"    [Strict Dedup] Đã lập danh sách {len(seen_urls):,} URL từng xuất hiện ở 3 nguồn Train để loại bỏ hoàn toàn trùng lặp!")

    def run_url_ext(title, urls_series, labels_series, max_n=60_000):
        df_u = pd.DataFrame({"url": urls_series.astype(str), "y": labels_series.astype(int)}).dropna()
        norm_check = df_u["url"].str.lower().str.replace(r"^https?://(www\.)?", "", regex=True).str.rstrip("/")
        unseen_mask = ~norm_check.isin(seen_urls)
        overlap_pct = (1.0 - unseen_mask.mean()) * 100.0
        df_u = df_u[unseen_mask].drop_duplicates(subset=["url"]).reset_index(drop=True)
        print(f"    [Dedup {title[:25]}...] Trùng với tập Train: {overlap_pct:.1f}% -> Giữ lại 100% URL MỚI: {len(df_u):,} URL")
        if len(df_u) > max_n:
            df_u = df_u.sample(n=max_n, random_state=42).reset_index(drop=True)
        if df_u["y"].nunique() < 2 or len(df_u) < 50:
            print("    -> Không đủ 2 lớp sau khi lọc trùng!")
            return
        X_lex, norms, _, tlds = extract_url_batch(df_u["url"].tolist())
        X_lex["tld_phish_rate"] = [tld_map.get(t, g_rate) for t in tlds]
        X_lex["tfidf_char_prob"] = char_clf.predict_proba(char_vec.transform(norms))[:, 1]
        X_lex["tfidf_word_prob"] = word_clf.predict_proba(word_vec.transform(norms))[:, 1]
        probs = lgb_m.predict_proba(X_lex[fcols])[:, 1]
        eval_report(title, df_u["y"].values, probs, th)

    # 3A: kmack/Phishing_urls (kiểm tra cực nhãn trước!)
    km = pd.read_parquet(EXTERNAL_URLS["kmack_url_test"][1])
    print("    [Check polarity kmack] label=0 sample:", km[km["label"] == 0]["text"].head(3).tolist())
    print("    [Check polarity kmack] label=1 sample:", km[km["label"] == 1]["text"].head(3).tolist())
    # Trong kmack: linuxplanet.com là label=1 hay 0? Sẽ tự nhận diện theo google/wiki/linuxplanet!
    km_lbl0_str = " ".join(km[km["label"] == 0]["text"].head(50).astype(str).tolist()).lower()
    km_lbl1_str = " ".join(km[km["label"] == 1]["text"].head(50).astype(str).tolist()).lower()
    # Nếu label=1 có nhiều wikipedia/youtube/facebook/amazon hơn thì label=1 là benign -> đảo nhãn!
    benign_markers = ["wikipedia.org", "youtube.com", "facebook.com", "github.com", "nytimes.com", "bbc.co.uk", "amazon.com"]
    c0 = sum(km_lbl0_str.count(m) for m in benign_markers)
    c1 = sum(km_lbl1_str.count(m) for m in benign_markers)
    km_y = (km["label"] == 0).astype(int) if c1 > c0 else (km["label"] == 1).astype(int)
    print(f"    [Auto-polarity kmack] benign markers in lbl0={c0}, lbl1={c1} -> Phishing is label={0 if c1 > c0 else 1}")
    run_url_ext("3A. EXTERNAL BENCHMARK: kmack/Phishing_urls (Test Split — 100% URL chưa từng thấy)", km["text"], km_y)

    # 3B: KaushiGihan/phishing-site-url
    ka = pd.read_parquet(EXTERNAL_URLS["kaushi_url_test"][1])
    print("    [Check polarity Kaushi] label=0 sample:", ka[ka["labels"] == 0]["text"].head(3).tolist())
    print("    [Check polarity Kaushi] label=1 sample:", ka[ka["labels"] == 1]["text"].head(3).tolist())
    run_url_ext("3B. EXTERNAL BENCHMARK: KaushiGihan/phishing-site-url (Test Split — 100% URL chưa từng thấy)", ka["text"], ka["labels"])

    # 3C: CyRadar Vietnam Phishing Detection Dataset (hãng an ninh mạng Việt Nam!)
    cy = pd.read_parquet(EXTERNAL_URLS["cyradar_vn_url"][1])
    print(f"    [Check polarity CyRadar VN] shape={cy.shape}, dist={cy['label'].value_counts().to_dict()}")
    print("    [Check polarity CyRadar VN] label=0 sample:", cy[cy["label"] == 0]["text"].head(3).tolist())
    print("    [Check polarity CyRadar VN] label=1 sample:", cy[cy["label"] == 1]["text"].head(3).tolist())
    run_url_ext("3C. EXTERNAL BENCHMARK VIỆT NAM: CyRadar Phishing Detection Dataset (60,000 URL mới)", cy["text"], cy["label"], max_n=60_000)


if __name__ == "__main__":
    download_all()
    test_paysim_external()
    test_sparkov_external()
    test_url_external()

