import os
import re
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import joblib
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
os.makedirs(REAL_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))

from train_phiusiil import extract_url_batch
from train_vesta_real import compute_uid_rolling_velocity


def fetch_text(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


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


def clean_norm(u):
    u = u.strip().lower()
    for pref in ("https://", "http://"):
        if u.startswith(pref):
            u = u[len(pref):]
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def fetch_hn_urls(limit=80):
    ids = json.loads(fetch_text("https://hacker-news.firebaseio.com/v0/topstories.json"))[:limit]

    def get_one(item_id):
        try:
            d = json.loads(fetch_text(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=6))
            return d.get("url")
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        urls = [u for u in ex.map(get_one, ids) if u and u.startswith("http")]
    return list(dict.fromkeys(urls))


def fetch_vn_rss_urls():
    urls = []
    rss_list = [
        "https://vnexpress.net/rss/tin-moi-nhat.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://tuoitre.vn/rss/tin-moi-nhat.rss",
        "https://dantri.com.vn/rss/home.rss",
    ]
    for r_url in rss_list:
        try:
            xml = fetch_text(r_url, timeout=10)
            found = re.findall(r"<link>\s*(https?://[^<\s]+)\s*</link>", xml)
            for u in found:
                # Chỉ lấy link bài báo sâu (không lấy trang chủ rss)
                if len(u) > 30 and not u.endswith(".rss"):
                    urls.append(u.strip())
        except Exception as e:
            print(f"  [Warn RSS] {r_url}: {e}")
    return list(dict.fromkeys(urls))


def test_extra_live_urls():
    print("=" * 88)
    print("EXTRA REAL-WORLD LIVE TEST: 4 NGUỒN DỮ LIỆU THẬT HOÀN TOÀN MỚI HÔM NAY (0% SYNTHETIC)")
    print("=" * 88)
    b = joblib.load(os.path.join(MODEL_DIR, "phiusiil_bundle.joblib"))
    th = b["threshold"]
    seen = load_seen_train_urls()
    rng = np.random.default_rng(42)

    def score_urls(url_list):
        X_lex, norms, _, tlds = extract_url_batch(url_list)
        X_lex["tld_phish_rate"] = [b["tld_rate_map"].get(t, b["global_rate"]) for t in tlds]
        X_lex["tfidf_char_prob"] = b["char_clf"].predict_proba(b["char_vec"].transform(norms))[:, 1]
        X_lex["tfidf_word_prob"] = b["word_clf"].predict_proba(b["word_vec"].transform(norms))[:, 1]
        return b["lgb_model"].predict_proba(X_lex[b["feature_cols"]])[:, 1]

    # 1. CERT.pl National Cybersecurity Center (Poland) — LIVE phishing domains today
    print("[1/4] Đang tải CERT.pl National CERT Live Phishing Domains...")
    cert_txt = fetch_text("https://hole.cert.pl/domains/v2/domains.txt", timeout=25)
    cert_raw = [x.strip() for x in cert_txt.splitlines() if x.strip() and not x.startswith("#") and "." in x]
    cert_new = ["https://" + d for d in cert_raw if clean_norm(d) not in seen]
    cert_sample = list(rng.choice(cert_new, size=min(10_000, len(cert_new)), replace=False))
    p_cert = score_urls(cert_sample)
    print(f"  -> CERT.pl tổng sống hôm nay: {len(cert_raw):,} tên miền | Mới 100% chưa thấy: {len(cert_new):,} -> Lấy mẫu {len(cert_sample):,}")
    print(f"  -> [KẾT QUẢ CERT.pl LIVE]: Bắt trúng {(p_cert>=th).mean()*100:.2f}% ({int((p_cert>=th).sum()):,}/{len(cert_sample):,}) | Điểm rủi ro TB: {p_cert.mean()*100:.1f}%")

    # 2. URLhaus abuse.ch Online Active Malicious URLs
    print("\n[2/4] Đang tải URLhaus (abuse.ch) Online Active Malicious URLs...")
    uh_txt = fetch_text("https://malware-filter.gitlab.io/urlhaus-filter/urlhaus-filter-online.txt", timeout=25)
    uh_raw = []
    for line in uh_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        # Một số dòng dạng domain hoặc domain/path
        if "/" in line or "." in line:
            u = line if line.startswith("http") else ("http://" + line)
            uh_raw.append(u)
    uh_new = [u for u in uh_raw if clean_norm(u) not in seen]
    uh_sample = list(rng.choice(uh_new, size=min(10_000, len(uh_new)), replace=False))
    p_uh = score_urls(uh_sample)
    print(f"  -> URLhaus tổng sống hôm nay: {len(uh_raw):,} URL/Host | Mới 100%: {len(uh_new):,} -> Lấy mẫu {len(uh_sample):,}")
    print(f"  -> [KẾT QUẢ URLhaus LIVE]: Bắt trúng {(p_uh>=th).mean()*100:.2f}% ({int((p_uh>=th).sum()):,}/{len(uh_sample):,}) | Điểm rủi ro TB: {p_uh.mean()*100:.1f}%")

    # 3. HackerNews Real Deep Benign URLs TODAY (bài test False Positive CỰC GẮT!)
    print("\n[3/4] Đang kéo trực tiếp bài viết thật hôm nay từ HackerNews (URL sâu, nhiều dấu -, query, path)...")
    hn_urls = fetch_hn_urls(90)
    hn_new = [u for u in hn_urls if clean_norm(u) not in seen]
    p_hn = score_urls(hn_new)
    hn_safe_rate = (p_hn < th).mean() * 100.0
    print(f"  -> HackerNews LIVE hôm nay: {len(hn_new)} URL sâu hoàn toàn mới")
    print(f"  -> [KẾT QUẢ HACKERNEWS DEEP BENIGN]: Nhận đúng AN TOÀN = {hn_safe_rate:.2f}% ({int((p_hn<th).sum())}/{len(hn_new)}) | Báo nhầm (FPR) = {100-hn_safe_rate:.2f}%")
    fp_hn = [(u, p) for u, p in zip(hn_new, p_hn) if p >= th]
    if fp_hn:
        print("     * Ví dụ các URL thật trên HackerNews bị báo nhầm (để thấy điểm yếu thật!):")
        for u, p in sorted(fp_hn, key=lambda x: -x[1])[:5]:
            print(f"       - [Báo nhầm {p*100:5.1f}%] {u}")

    # 4. Báo chí Việt Nam LIVE RSS hôm nay (VNExpress + TuoiTre + DanTri — URL tiếng Việt có nhiều '-' + số ID + '.html'!)
    print("\n[4/4] Đang kéo trực tiếp link bài báo thật HÔM NAY từ VNExpress, Tuổi Trẻ, Dân Trí (RSS LIVE)...")
    vn_rss_urls = fetch_vn_rss_urls()
    vn_new = [u for u in vn_rss_urls if clean_norm(u) not in seen]
    p_vn = score_urls(vn_new)
    vn_safe_rate = (p_vn < th).mean() * 100.0
    print(f"  -> Báo chí VN LIVE hôm nay: {len(vn_new)} link bài báo sâu (nhiều gạch ngang + số ID + .html/.htm)")
    print(f"  -> [KẾT QUẢ BÁO CHÍ VN DEEP BENIGN]: Nhận đúng AN TOÀN = {vn_safe_rate:.2f}% ({int((p_vn<th).sum())}/{len(vn_new)}) | Báo nhầm (FPR) = {100-vn_safe_rate:.2f}%")
    fp_vn = [(u, p) for u, p in zip(vn_new, p_vn) if p >= th]
    if fp_vn:
        print("     * Ví dụ link báo VN bị báo nhầm (để đánh giá điểm yếu thật!):")
        for u, p in sorted(fp_vn, key=lambda x: -x[1])[:5]:
            print(f"       - [Báo nhầm {p*100:5.1f}%] {u}")

    # Tổng hợp AUC thực tế: CERT.pl + URLhaus (Phishing thật sống hôm nay) vs HackerNews + Báo VN (Deep Benign thật sống hôm nay!)
    y_hard = np.concatenate([np.ones(len(cert_sample)), np.ones(len(uh_sample)), np.zeros(len(hn_new)), np.zeros(len(vn_new))])
    p_hard = np.concatenate([p_cert, p_uh, p_hn, p_vn])
    print("\n>>> TỔNG HỢP HARD LIVE TEST (20,000 tấn công sống CERT.pl+URLhaus vs Deep Benign thật HackerNews+Báo VN):")
    print(f"    ROC-AUC = {roc_auc_score(y_hard, p_hard):.4f} | PR-AUC = {average_precision_score(y_hard, p_hard):.4f}")


def test_vesta_robustness():
    print("\n" + "=" * 88)
    print("EXTRA REAL-WORLD STRESS TEST CHO MODEL THANH TOÁN THẬT IEEE-CIS VESTA")
    print("=" * 88)
    b = joblib.load(os.path.join(MODEL_DIR, "vesta_real_bundle.joblib"))
    lgb_m = b["lgb_model"]
    fcols = b["feature_cols"]
    maps = b["oof_maps"]
    g_rate = b["global_rate"]
    prod_stats = b["prod_stats"]
    th = b["threshold"]

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
    test_df = df.iloc[int(n * 0.85):].copy().reset_index(drop=True)
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

    # Test A: Độ trôi thời gian (Time-Decay Stability) chia tập Test tương lai thành 4 tuần liên tiếp!
    print("\n[STRESS TEST A] ĐỘ ỔN ĐỊNH THEO THỜI GIAN TRÔI (4 TUẦN TƯƠNG LAI LIÊN TIẾP TRONG TẬP TEST):")
    chunks = np.array_split(np.arange(len(test_df)), 4)
    for idx_w, idxs in enumerate(chunks, 1):
        yt, pt = y_test[idxs], probs[idxs]
        yp = (pt >= th).astype(int)
        print(
            f"  * Tuần tương lai #{idx_w} ({len(idxs):,} GD | {int(yt.sum()):,} fraud): "
            f"ROC-AUC={roc_auc_score(yt, pt):.4f} | PR-AUC={average_precision_score(yt, pt):.4f} | "
            f"F1={f1_score(yt, yp):.4f} | Rec={recall_score(yt, yp)*100:.1f}%"
        )

    # Test B: Stress Test BỊ CHE DỮ LIỆU (Mask 100% toàn bộ cột vesta_v_block_* & vesta_v_ind_* thành NaN!)
    X_masked = X_test.copy()
    v_cols = [c for c in fcols if c.startswith("vesta_v_")]
    X_masked[v_cols] = np.nan
    p_masked = lgb_m.predict_proba(X_masked)[:, 1]
    print("\n[STRESS TEST B] CHE TRẮNG 100% TOÀN BỘ 10 CỘT VESTA V-BLOCK (GIẢ LẬP THIẾU TELEMETRY):")
    print(
        f"  * Full features (70 cột)        : ROC-AUC={roc_auc_score(y_test, probs):.4f} | PR-AUC={average_precision_score(y_test, probs):.4f}"
    )
    print(
        f"  * Khi xóa trắng 10 cột V-block  : ROC-AUC={roc_auc_score(y_test, p_masked):.4f} | PR-AUC={average_precision_score(y_test, p_masked):.4f}"
    )

    # Test C: Stress Test MẤT LỊCH SỬ THẺ (Đặt toàn bộ OOF target-encoding về global_rate trung bình!)
    X_no_oof = X_test.copy()
    for c in [c for c in fcols if c.endswith("_oof_risk")]:
        X_no_oof[c] = g_rate
    p_no_oof = lgb_m.predict_proba(X_no_oof)[:, 1]
    print("\n[STRESS TEST C] XÓA TRẮNG TOÀN BỘ LỊCH SỬ RỦI RO OOF (card1/card2/addr1/email/device = trung bình):")
    print(
        f"  * Không có bất kỳ lịch sử nào   : ROC-AUC={roc_auc_score(y_test, p_no_oof):.4f} | PR-AUC={average_precision_score(y_test, p_no_oof):.4f}"
    )


if __name__ == "__main__":
    test_extra_live_urls()
    test_vesta_robustness()

