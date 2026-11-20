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
REAL_DIR = os.path.join(DATA_DIR, "real_world_test")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(REAL_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from train_phiusiil import extract_url_batch

SOURCES = {
    "openphish_live": (
        "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt",
        os.path.join(REAL_DIR, "openphish_live.txt"),
    ),
    "phishing_db_active": (
        "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt",
        os.path.join(REAL_DIR, "phishing_db_active.txt"),
    ),
    "opendns_top_real": (
        "https://raw.githubusercontent.com/opendns/public-domain-lists/master/opendns-top-domains.txt",
        os.path.join(REAL_DIR, "opendns_top_real.txt"),
    ),
    "vesta_real_ieee_cis": (
        "https://huggingface.co/datasets/t8nisho/vesta-fraud-data/resolve/main/vesta_condensed_final.csv",
        os.path.join(REAL_DIR, "vesta_real_ieee_cis.csv"),
    ),
}

REAL_VN_SAFE_URLS = [
    "https://momo.vn",
    "https://vietcombank.com.vn/ca-nhan",
    "https://techcombank.com/khach-hang-ca-nhan",
    "https://bidv.com.vn/vn/ca-nhan",
    "https://mbbank.com.vn",
    "https://vpbank.com.vn",
    "https://acb.com.vn",
    "https://tpb.vn",
    "https://zalopay.vn",
    "https://vnpay.vn",
    "https://vnexpress.net/kinh-doanh",
    "https://tuoitre.vn/kinh-doanh.htm",
    "https://dantri.com.vn/kinh-doanh.htm",
    "https://baochinhphu.vn",
    "https://dichvucong.gov.vn",
    "https://tiki.vn",
    "https://lazada.vn",
    "https://fptshop.com.vn",
    "https://thegioididong.com",
    "https://viettel.vn",
]


def download_real_sources():
    print("[0/3] Đang tải 100% DỮ LIỆU THỰC TẾ NGOÀI ĐỜI (NON-SYNTHETIC & LIVE FEEDS)...")
    for key, (url, path) in SOURCES.items():
        if os.path.exists(path) and os.path.getsize(path) > 5_000:
            print(f"  [OK] Đã có sẵn {key}: {os.path.basename(path)} ({os.path.getsize(path)/(1024*1024):.2f} MB)")
            continue
        t0 = time.time()
        print(f"  -> Đang tải {key} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
        print(f"     Xong ({os.path.getsize(path)/(1024*1024):.2f} MB) trong {time.time()-t0:.1f}s")


def load_seen_train_urls():
    seen = set()
    for fn, col in [
        ("phiusiil_raw.parquet", "phiusiil_url"),
        ("mitake_urls.parquet", "url"),
        ("pirocheto_urls.parquet", "url"),
    ]:
        fp = os.path.join(DATA_DIR, fn)
        if os.path.exists(fp):
            s = (
                pd.read_parquet(fp, columns=[col])[col]
                .astype(str)
                .str.lower()
                .str.replace(r"^https?://(www\.)?", "", regex=True)
                .str.rstrip("/")
            )
            seen.update(s.tolist())
    return seen


def eval_block(title, y_true, y_prob, th):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= th).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / max(fp + tn, 1)
    print(f"\n>>> {title}")
    print(f"    Samples = {len(y_true):,} | Real Attacks/Frauds = {int(y_true.sum()):,} ({y_true.mean()*100:.2f}%)")
    print(f"    ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f} | F1 = {f1:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}% | FPR = {fpr*100:.2f}%")
    print(f"    Confusion Matrix: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,} (Threshold={th:.4f})")


def test_real_live_urls():
    print("\n" + "=" * 84)
    print("REAL-WORLD TEST 1: MODULE 3 (URL PHISHING) TRÊN LIVE PHISHING FEEDS HÔM NAY (0% SYNTHETIC)")
    print("=" * 84)
    b = joblib.load(os.path.join(MODEL_DIR, "phiusiil_bundle.joblib"))
    seen = load_seen_train_urls()

    def clean_norm(u):
        u = u.strip().lower()
        for pref in ("https://", "http://"):
            if u.startswith(pref):
                u = u[len(pref):]
        if u.startswith("www."):
            u = u[4:]
        return u.rstrip("/")

    # 1. OpenPhish LIVE today
    with open(SOURCES["openphish_live"][1], "r", encoding="utf-8", errors="ignore") as f:
        op_raw = [x.strip() for x in f if x.strip().startswith("http")]
    op_new = [u for u in op_raw if clean_norm(u) not in seen]

    # 2. Phishing.Database ACTIVE live links (lấy 10,000 link mới nhất chưa từng xuất hiện trong Train)
    with open(SOURCES["phishing_db_active"][1], "r", encoding="utf-8", errors="ignore") as f:
        pdb_raw = [x.strip() for x in f if x.strip().startswith("http")]
    rng = np.random.default_rng(42)
    pdb_unseen = [u for u in pdb_raw if clean_norm(u) not in seen]
    if len(pdb_unseen) > 10_000:
        pdb_sample = list(rng.choice(pdb_unseen, size=10_000, replace=False))
    else:
        pdb_sample = pdb_unseen

    # 3. OpenDNS Top Real Legitimate Domains (thật 100%, lọc bỏ domain từng có trong Train)
    with open(SOURCES["opendns_top_real"][1], "r", encoding="utf-8", errors="ignore") as f:
        legit_raw = ["https://" + x.strip() for x in f if x.strip() and not x.startswith("#")]
    legit_unseen = [u for u in legit_raw if clean_norm(u) not in seen]

    print(f"  [Live Feed Stats — 0% trùng Train]:")
    print(f"    - OpenPhish LIVE hôm nay          : {len(op_raw):,} tổng -> {len(op_new):,} link hoàn toàn MỚI")
    print(f"    - Phishing.Database ACTIVE hôm nay: {len(pdb_raw):,} tổng -> lấy mẫu {len(pdb_sample):,} link hoàn toàn MỚI")
    print(f"    - OpenDNS Top Real Legit Domains  : {len(legit_raw):,} tổng -> {len(legit_unseen):,} domain hoàn toàn MỚI")

    def score_urls(url_list):
        X_lex, norms, _, tlds = extract_url_batch(url_list)
        X_lex["tld_phish_rate"] = [b["tld_rate_map"].get(t, b["global_rate"]) for t in tlds]
        X_lex["tfidf_char_prob"] = b["char_clf"].predict_proba(b["char_vec"].transform(norms))[:, 1]
        X_lex["tfidf_word_prob"] = b["word_clf"].predict_proba(b["word_vec"].transform(norms))[:, 1]
        return b["lgb_model"].predict_proba(X_lex[b["feature_cols"]])[:, 1]

    p_op = score_urls(op_new)
    p_pdb = score_urls(pdb_sample)
    p_legit = score_urls(legit_unseen)
    p_vn = score_urls(REAL_VN_SAFE_URLS)

    th = b["threshold"]
    print(f"\n  [Recall trên LIVE OpenPhish hôm nay ({len(op_new)} link tấn công thật)]: {(p_op >= th).mean()*100:.2f}% ({int((p_op >= th).sum())}/{len(op_new)})")
    print(f"  [Recall trên LIVE Phishing.Database ACTIVE ({len(pdb_sample):,} link tấn công thật)]: {(p_pdb >= th).mean()*100:.2f}% ({int((p_pdb >= th).sum()):,}/{len(pdb_sample):,})")
    print(f"  [Độ chính xác trên 20 website thật Việt Nam (Ngân hàng/Ví/Báo chí/TMĐT)]: {(p_vn < th).mean()*100:.2f}% ({int((p_vn < th).sum())}/{len(REAL_VN_SAFE_URLS)} an toàn)")
    for u, p in zip(REAL_VN_SAFE_URLS, p_vn):
        if p >= th:
            print(f"      -> Báo nhầm VN: {u} ({p*100:.1f}%)")

    # Đánh giá tổng hợp: OpenPhish Live + Phishing.Database Live vs OpenDNS Top Real Legit
    y_all = np.concatenate([np.ones(len(op_new)), np.ones(len(pdb_sample)), np.zeros(len(legit_unseen))])
    p_all = np.concatenate([p_op, p_pdb, p_legit])
    eval_block("1A. LIVE INTERNET BENCHMARK (OpenPhish + Phishing.Database ACTIVE vs OpenDNS Top Real — 0% Synthetic)", y_all, p_all, th)

    y_op_legit = np.concatenate([np.ones(len(op_new)), np.zeros(len(legit_unseen))])
    p_op_legit = np.concatenate([p_op, p_legit])
    eval_block("1B. LIVE OPENPHISH TODAY vs OPENDNS TOP REAL WEBSITES", y_op_legit, p_op_legit, th)


def compute_vesta_rolling(df):
    n = len(df)
    c1h = np.zeros(n, dtype=np.float32)
    c24h = np.zeros(n, dtype=np.float32)
    s24h = np.zeros(n, dtype=np.float32)
    cc = df["card1"].fillna(-1).values
    ts = df["TransactionDT"].values
    am = df["TransactionAmt"].values
    order = np.argsort(cc, kind="mergesort")
    cc_s, ts_s, am_s = cc[order], ts[order], am[order]
    ch = np.flatnonzero(np.r_[True, cc_s[1:] != cc_s[:-1], True])
    for k in range(len(ch) - 1):
        st, en = ch[k], ch[k + 1]
        t_arr, a_arr = ts_s[st:en], am_s[st:en]
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
    return c1h, c24h, s24h


def test_real_vesta_ieee_cis():
    print("\n" + "=" * 84)
    print("REAL-WORLD TEST 2: MODULE 2 (SPARKOV) TRÊN GIAO DỊCH THẬT IEEE-CIS VESTA (0% SYNTHETIC!)")
    print("=" * 84)
    usecols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "card6", "dist1"]
    df = pd.read_csv(SOURCES["vesta_real_ieee_cis"][1], usecols=usecols)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print(f"  [IEEE-CIS Vesta Real Data] Tổng giao dịch THẬT: {len(df):,} | Gian lận THẬT: {int(df['isFraud'].sum()):,} ({df['isFraud'].mean()*100:.2f}%)")

    b = joblib.load(os.path.join(MODEL_DIR, "sparkov_bundle.joblib"))
    lgb_m = b["lgb_model"]
    th = b["threshold"]
    fcols = b["feature_cols"]
    cat_map = b["cat_map"]
    cat_stats = b["cat_stats"]
    cat_fraud_rate = b["cat_fraud_rate"]
    global_rate = b["global_rate"]

    # Map IEEE-CIS ProductCD (W=Web/Shopping, C=Digital/Misc, R=Retail/Travel, H=Hotel/Travel, S=Service) -> Sparkov categories
    prod_to_cat = {"W": "shopping_net", "C": "misc_net", "R": "shopping_pos", "H": "travel", "S": "misc_pos"}
    mapped_cat = df["ProductCD"].map(prod_to_cat).fillna("misc_net")

    x = pd.DataFrame(index=df.index)
    x["amt"] = df["TransactionAmt"].astype(float)
    x["amt_log"] = np.log1p(np.maximum(x["amt"], 0.0))
    x["category_code"] = mapped_cat.map(cat_map).fillna(0).astype(int)
    x["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(int)
    x["hour_sin"] = np.sin(2.0 * np.pi * x["hour"] / 24.0)
    x["hour_cos"] = np.cos(2.0 * np.pi * x["hour"] / 24.0)
    x["is_night"] = x["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    x["day_of_week"] = ((df["TransactionDT"] // 86400) % 7).astype(int)
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)
    x["cardholder_age"] = 38.0
    x["gender_code"] = 0
    x["city_pop_log"] = np.log1p(300_000.0)
    x["distance_km"] = df["dist1"].fillna(25.0).astype(float).clip(lower=0.0)

    grp = df.groupby("card1", sort=False)
    prev_t = grp["TransactionDT"].shift(1)
    prev2_t = grp["TransactionDT"].shift(2)
    prev_amt = grp["TransactionAmt"].shift(1)
    prev_cat = mapped_cat.groupby(df["card1"], sort=False).shift(1)

    x["time_since_last_tx_sec"] = (df["TransactionDT"] - prev_t).fillna(86400.0).clip(lower=1.0)
    x["time_since_last_2tx_sec"] = (df["TransactionDT"] - prev2_t).fillna(172800.0).clip(lower=1.0)
    x["dist_from_last_merch_km"] = x["distance_km"]
    x["travel_speed_kmh"] = np.clip(x["distance_km"] / (x["time_since_last_tx_sec"] / 3600.0 + 1e-4), 0.0, 5000.0)

    c1h, c24h, s24h = compute_vesta_rolling(df)
    x["tx_count_last_1h"] = c1h
    x["tx_count_last_24h"] = c24h
    x["amt_sum_last_24h"] = s24h
    x["amt_vs_prev_ratio"] = np.clip(x["amt"] / (prev_amt.fillna(x["amt"]) + 1.0), 0.0, 100.0)
    x["same_cat_as_prev"] = (mapped_cat == prev_cat).astype(int)

    x["user_tx_count_prev"] = grp.cumcount()
    cum_prev = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]
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

    probs = lgb_m.predict_proba(x[fcols])[:, 1]
    eval_block("2A. ZERO-SHOT: Sparkov Model (Synthetic) -> 590,540 GIAO DỊCH THẬT IEEE-CIS Vesta (Real-World)", df["isFraud"], probs, th)


if __name__ == "__main__":
    download_real_sources()
    test_real_live_urls()
    test_real_vesta_ieee_cis()

