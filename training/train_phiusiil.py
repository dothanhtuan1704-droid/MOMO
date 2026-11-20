import os
import re
import math
import time
from urllib.parse import urlparse
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
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

PHIUSIIL_PATH = os.path.join(DATA_DIR, "phiusiil_raw.parquet")
MITAKE_PATH = os.path.join(DATA_DIR, "mitake_urls.parquet")
PIRO_PATH = os.path.join(DATA_DIR, "pirocheto_urls.parquet")

RISKY_TLDS = {
    "xyz", "top", "club", "tk", "ml", "ga", "cf", "gq", "buzz", "info", "online",
    "site", "vip", "icu", "shop", "cc", "pw", "work", "click", "link", "live",
    "rest", "cfd", "sbs", "monster", "pro", "su", "ws", "cyou", "loan", "ru", "cn", "pl", "br"
}
TRUSTED_TLDS = {"gov", "edu", "mil", "org", "vn", "com", "net", "uk", "de", "jp", "fr", "ca", "au"}
BRAND_NAMES = (
    "paypal", "apple", "amazon", "microsoft", "google", "facebook", "instagram", "netflix",
    "chase", "wellsfargo", "citibank", "hsbc", "binance", "coinbase",
    "momo", "vietcombank", "vcb", "bidv", "techcombank", "mbbank", "vpbank", "acb",
    "vietinbank", "agribank", "tpbank", "sacombank", "zalopay", "vnpay", "shopee", "lazada", "tiki"
)
PHISH_ACTION_WORDS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "wallet", "support", "security", "password", "credential", "billing", "invoice",
    "free", "gift", "bonus", "reward", "claim", "urgent", "suspend", "unlock",
    "xacminh", "baomat", "nhanqua", "trian", "khoiphuc", "dangnhap", "otp", "khuyenmai", "wp-includes", "wp-content"
)
FREE_HOSTING = (
    "firebaseapp.com", "web.app", "pages.dev", "netlify.app", "vercel.app", "herokuapp.com",
    "blogspot.com", "wordpress.com", "000webhostapp.com", "wixsite.com", "weebly.com",
    "duckdns.org", "ngrok.io", "github.io", "gitlab.io", "azurewebsites.net", "s3.amazonaws.com",
    "repl.co", "yolasite.com", "000webhost.com", "sites.google.com", "storage.cloud.google.com"
)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy", "shorturl.at", "x.co")
IP_REGEX = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
SCHEME_REGEX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
EXT_REGEX = re.compile(r"\.(php|asp|aspx|jsp|exe|apk|zip|scr|htm|html)(\?|$)")
CONSEC_DIGITS_REGEX = re.compile(r"\d+")

FEATURE_NAMES_VI = {
    "url_len": "Tổng độ dài URL (sau chuẩn hóa)",
    "domain_len": "Độ dài tên miền (Domain)",
    "path_len": "Độ dài phần đường dẫn phụ (Path)",
    "query_len": "Độ dài tham số truy vấn (Query string)",
    "path_to_url_ratio": "Tỷ lệ độ dài đường dẫn phụ trên tổng URL",
    "has_ip": "Dùng trực tiếp địa chỉ IP thay vì tên miền hợp lệ",
    "subdomain_count": "Số tầng tên miền phụ (Subdomain) bất thường",
    "num_dots_domain": "Số dấu chấm (.) trong tên miền",
    "num_hyphens_domain": "Số dấu gạch ngang (-) trong tên miền (dấu hiệu giả mạo thương hiệu)",
    "num_hyphens_url": "Tổng số dấu gạch ngang (-) trong toàn URL",
    "num_slashes": "Số tầng thư mục (/) trong đường dẫn",
    "num_digits_domain": "Số chữ số bất thường trong tên miền",
    "max_consec_digits": "Chuỗi chữ số liên tiếp dài nhất trong URL (mã ngẫu nhiên)",
    "digit_ratio_url": "Tỷ lệ chữ số trên toàn đường dẫn",
    "vowel_ratio_domain": "Tỷ lệ nguyên âm trong tên miền (thấp bất thường ở tên miền sinh tự động)",
    "special_chars": "Số ký tự đặc biệt (@, ?, =, %, &, _)",
    "has_at": "Chứa ký tự '@' đánh lừa trình duyệt",
    "has_punycode": "Chứa mã hóa Punycode (xn--) giả mạo ký tự Unicode",
    "is_shortener": "Dùng dịch vụ rút gọn link che giấu đích đến",
    "is_free_hosting": "Chạy trên nền tảng hosting miễn phí hay bị lợi dụng tạo trang giả",
    "is_risky_tld": "Đuôi tên miền thuộc nhóm rủi ro cao (.xyz, .top, .vip, .tk...)",
    "is_trusted_tld": "Đuôi tên miền phổ biến (.com, .vn, .org, .edu, .gov...)",
    "tld_phish_rate": "Tỷ lệ lừa đảo lịch sử của đuôi tên miền (TLD Risk Rate)",
    "brand_impersonation": "Giả mạo thương hiệu/ngân hàng (chứa tên hãng kèm từ lạ/subdomain/TLD giả)",
    "phish_action_kw_count": "Chứa từ khóa thao túng đăng nhập/xác minh/nhận quà/OTP",
    "has_suspicious_ext": "Đuôi file .php/.htm/.exe/.apk đáng ngờ trong đường dẫn",
    "url_entropy": "Độ hỗn loạn ký tự (Shannon Entropy) của đường dẫn",
    "domain_entropy": "Độ hỗn loạn ký tự của tên miền (tên miền sinh ngẫu nhiên)",
    "tfidf_char_prob": "Điểm nhận diện mẫu chuỗi ký tự N-gram (Char TF-IDF AI)",
    "tfidf_word_prob": "Điểm nhận diện ngữ nghĩa đường dẫn (Token/Word TF-IDF AI)",
}
LEXICAL_COLS = [
    c for c in FEATURE_NAMES_VI.keys()
    if c not in ("tld_phish_rate", "tfidf_char_prob", "tfidf_word_prob")
]
FEATURE_COLS = list(FEATURE_NAMES_VI.keys())


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def normalize_url_debiased(raw_url: str):
    url = str(raw_url).strip()
    parse_target = url if SCHEME_REGEX.match(url) else ("http://" + url)
    try:
        parsed = urlparse(parse_target)
    except Exception:
        parsed = urlparse("http://invalid.local")

    netloc = (parsed.netloc or "").lower().split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = (parsed.path or "").rstrip("/")
    query = parsed.query or ""
    norm_str = netloc + path + (("?" + query) if query else "")
    return netloc, path, query, norm_str


def parse_domain_parts(netloc: str):
    parts = [p for p in netloc.split(".") if p]
    if not parts:
        return "", "", 0, ""
    tld = parts[-1]
    if len(parts) >= 3 and parts[-2] in {"com", "co", "org", "net", "edu", "gov", "ac"} and len(tld) == 2:
        root_label = parts[-3]
        sub_count = max(0, len(parts) - 3)
        sub_str = ".".join(parts[:-3])
    elif len(parts) >= 2:
        root_label = parts[-2]
        sub_count = max(0, len(parts) - 2)
        sub_str = ".".join(parts[:-2])
    else:
        root_label = parts[0]
        sub_count = 0
        sub_str = ""
    return root_label, tld, sub_count, sub_str


def extract_url_features_single(raw_url: str):
    netloc, path, query, norm_str = normalize_url_debiased(raw_url)
    norm_lower = norm_str.lower()
    root_label, tld, subdomain_count, sub_str = parse_domain_parts(netloc)

    brand_impersonation = 0
    for b in BRAND_NAMES:
        in_root = b in root_label
        exact_root = (root_label == b)
        in_sub = b in sub_str
        in_path = b in path.lower()
        if exact_root and tld in RISKY_TLDS:
            brand_impersonation = 1
            break
        if (in_root and not exact_root) or in_sub or (in_path and not exact_root):
            brand_impersonation = 1
            break

    # Chỉ tìm từ khóa hành động lừa đảo ở phần subdomain + path + query + non-exact root
    kw_search_area = (sub_str + " " + path.lower() + " " + query.lower() + " " + ("" if root_label in BRAND_NAMES else root_label))
    kw_count = sum(1 for kw in PHISH_ACTION_WORDS if kw in kw_search_area)

    num_digits_url = sum(c.isdigit() for c in norm_str)
    num_digits_dom = sum(c.isdigit() for c in netloc)
    m_digs = CONSEC_DIGITS_REGEX.findall(norm_str)
    max_consec_digits = max((len(x) for x in m_digs), default=0)
    dom_alpha = [c for c in netloc if c.isalpha()]
    vowel_ratio_dom = sum(c in "aeiou" for c in dom_alpha) / max(len(dom_alpha), 1)

    special_chars = sum(norm_str.count(ch) for ch in "@?=%&_~")
    is_free_host = 1 if any(netloc == fh or netloc.endswith("." + fh) for fh in FREE_HOSTING) else 0
    has_ext = 1 if EXT_REGEX.search(norm_lower) else 0
    u_len = max(len(norm_str), 1)

    feats = (
        len(norm_str),
        len(netloc),
        len(path),
        len(query),
        len(path) / u_len,
        1 if IP_REGEX.match(netloc) else 0,
        subdomain_count,
        netloc.count("."),
        netloc.count("-"),
        norm_str.count("-"),
        path.count("/"),
        num_digits_dom,
        max_consec_digits,
        num_digits_url / u_len,
        vowel_ratio_dom,
        special_chars,
        1 if "@" in norm_str else 0,
        1 if "xn--" in netloc else 0,
        1 if any(sh == netloc or netloc.endswith("." + sh) for sh in SHORTENERS) else 0,
        is_free_host,
        1 if tld in RISKY_TLDS else 0,
        1 if tld in TRUSTED_TLDS else 0,
        brand_impersonation,
        kw_count,
        has_ext,
        shannon_entropy(norm_lower),
        shannon_entropy(netloc),
    )
    return feats, norm_lower, netloc, tld


def extract_url_batch(urls):
    rows = []
    norms = []
    domains = []
    tlds = []
    for u in urls:
        f, n_str, dom, tld = extract_url_features_single(u)
        rows.append(f)
        norms.append(n_str)
        domains.append(dom)
        tlds.append(tld)
    df_lex = pd.DataFrame(rows, columns=LEXICAL_COLS, dtype=np.float32)
    return df_lex, norms, domains, tlds


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main():
    print("=" * 78)
    print("MODULE 3 (FULL 1.1M REAL-WORLD URLs): PHIUSIIL + FULL MITAKE + PIROCHETO")
    print("=" * 78)
    t_all = time.time()

    df1 = pd.read_parquet(PHIUSIIL_PATH)
    if "_keep_row" in df1.columns:
        df1 = df1[df1["_keep_row"] == True]
    df1 = pd.DataFrame({
        "url": df1["phiusiil_url"].astype(str),
        "is_phishing": (df1["label_str"].str.lower() != "legitimate").astype(int),
    })

    df2_raw = pd.read_parquet(MITAKE_PATH)
    df2 = pd.DataFrame({
        "url": df2_raw["url"].astype(str),
        "is_phishing": (df2_raw["label"].astype(int) == 1).astype(int),
    })

    df3_raw = pd.read_parquet(PIRO_PATH)
    df3 = pd.DataFrame({
        "url": df3_raw["url"].astype(str),
        "is_phishing": (df3_raw["status"].astype(str).str.lower() == "phishing").astype(int),
    })

    df = pd.concat([df1, df2, df3], ignore_index=True).dropna(subset=["url"])
    print(f"[1/4] Đã nạp toàn bộ 3 nguồn (chưa khử trùng): {len(df):,} URL")

    print("[2/4] Chuẩn hóa khử bias & trích xuất 27 đặc trưng cấu trúc URL cho >1.1 triệu URL...")
    t_ex = time.time()
    X_lex, norm_urls, domains, tlds = extract_url_batch(df["url"].tolist())
    df["norm_url"] = norm_urls
    df["domain_group"] = domains
    df["tld"] = tlds

    keep_mask = ~df["norm_url"].duplicated(keep="first")
    df = df[keep_mask].reset_index(drop=True)
    X_lex = X_lex[keep_mask.values].reset_index(drop=True)
    print(
        f"      -> Sau khử trùng lặp: {len(df):,} URL duy nhất ({time.time()-t_ex:.1f}s) | "
        f"Phishing: {int(df['is_phishing'].sum()):,} ({df['is_phishing'].mean()*100:.2f}%) | "
        f"Benign: {int((df['is_phishing']==0).sum()):,}"
    )

    # Domain-Grouped 70% Train / 15% Val / 15% Test (Zero Domain Leakage!)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, tmp_idx = next(gss1.split(df, df["is_phishing"], groups=df["domain_group"]))
    tmp_df = df.iloc[tmp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    va_rel, te_rel = next(gss2.split(tmp_df, tmp_df["is_phishing"], groups=tmp_df["domain_group"]))
    va_idx = tmp_idx[va_rel]
    te_idx = tmp_idx[te_rel]

    y_train = df["is_phishing"].values[tr_idx]
    y_val = df["is_phishing"].values[va_idx]
    y_test = df["is_phishing"].values[te_idx]

    # TLD Smoothed Target Encoding CHỈ học trên Train
    global_rate = float(y_train.mean())
    tld_tr = pd.DataFrame({"tld": df["tld"].iloc[tr_idx].values, "y": y_train})
    tld_agg = tld_tr.groupby("tld")["y"].agg(["count", "mean"])
    tld_rate_map = ((tld_agg["count"] * tld_agg["mean"] + 50 * global_rate) / (tld_agg["count"] + 50)).to_dict()

    print(
        f"[3/4] Huấn luyện Dual-NLP Out-Of-Fold Layer 1 (Char N-gram + Word/Token TF-IDF) "
        f"[Train={len(tr_idx):,} | Val={len(va_idx):,} | Test={len(te_idx):,}]..."
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=8,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vec = TfidfVectorizer(
        token_pattern=r"[a-z0-9]{2,}",
        ngram_range=(1, 2),
        min_df=6,
        max_features=80_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_char_tr = char_vec.fit_transform(df["norm_url"].iloc[tr_idx])
    X_char_va = char_vec.transform(df["norm_url"].iloc[va_idx])
    X_char_te = char_vec.transform(df["norm_url"].iloc[te_idx])

    X_word_tr = word_vec.fit_transform(df["norm_url"].iloc[tr_idx])
    X_word_va = word_vec.transform(df["norm_url"].iloc[va_idx])
    X_word_te = word_vec.transform(df["norm_url"].iloc[te_idx])

    oof_char = np.zeros(len(tr_idx), dtype=np.float32)
    oof_word = np.zeros(len(tr_idx), dtype=np.float32)
    dom_folds = df["domain_group"].iloc[tr_idx].astype("category").cat.codes.values % 3
    for fold in range(3):
        m_tr = dom_folds != fold
        m_va = dom_folds == fold
        c_clf = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")
        c_clf.fit(X_char_tr[m_tr], y_train[m_tr])
        oof_char[m_va] = c_clf.predict_proba(X_char_tr[m_va])[:, 1]

        w_clf = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")
        w_clf.fit(X_word_tr[m_tr], y_train[m_tr])
        oof_word[m_va] = w_clf.predict_proba(X_word_tr[m_va])[:, 1]

    char_clf = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")
    char_clf.fit(X_char_tr, y_train)
    word_clf = LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")
    word_clf.fit(X_word_tr, y_train)

    def assemble_X(idxs, c_prob, w_prob):
        part = X_lex.iloc[idxs].copy()
        part["tld_phish_rate"] = df["tld"].iloc[idxs].map(tld_rate_map).fillna(global_rate).values.astype(np.float32)
        part["tfidf_char_prob"] = c_prob.astype(np.float32)
        part["tfidf_word_prob"] = w_prob.astype(np.float32)
        return part[FEATURE_COLS]

    X_train = assemble_X(tr_idx, oof_char, oof_word)
    X_val = assemble_X(va_idx, char_clf.predict_proba(X_char_va)[:, 1], word_clf.predict_proba(X_word_va)[:, 1])
    X_test = assemble_X(te_idx, char_clf.predict_proba(X_char_te)[:, 1], word_clf.predict_proba(X_word_te)[:, 1])

    print("[4/4] Huấn luyện Layer 2 (LightGBM Hybrid URL Classifier trên ~770k URL)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=10,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
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

    print("\n" + "-" * 78)
    print(f"KẾT QUẢ FULL 1.1M URL ENGINE TRÊN TẬP TEST DOMAIN HOÀN TOÀN MỚI ({len(X_test):,} URL):")
    print(f"  * Best Threshold (từ Val) : {best_th:.4f} (Val F1 = {val_f1:.4f})")
    print(f"  * ROC-AUC                 : {roc_auc:.5f}")
    print(f"  * PR-AUC (Avg Precision)  : {pr_auc:.5f}")
    print(f"  * F1-Score                : {f1:.5f}")
    print(f"  * Precision (Độ chuẩn xác): {prec*100:.2f}%")
    print(f"  * Recall (Tỷ lệ bắt đúng) : {rec*100:.2f}%")
    print(f"  * False Positive Rate     : {fpr*100:.4f}%")
    print(f"  * Confusion Matrix        : TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print(f"  * Tổng thời gian chạy     : {time.time() - t_all:.1f}s")
    print("-" * 78)

    stress_urls = [
        ("SAFE", "https://momo.vn"),
        ("SAFE", "https://www.vietcombank.com.vn/ca-nhan"),
        ("SAFE", "https://shopee.vn/cart"),
        ("SAFE", "https://techcombank.com/khach-hang-ca-nhan"),
        ("SAFE", "https://google.com"),
        ("SAFE", "https://vnexpress.net/kinh-doanh"),
        ("PHISH", "http://momo-nhanqua-tri-an-2026.xyz/login/verify-account"),
        ("PHISH", "http://192.168.45.12/secure-bank-update/confirm.php?id=99182"),
        ("PHISH", "http://vietcombank-xacminh-baomat.top/signin/otp"),
        ("PHISH", "https://paypal-verify-account.firebaseapp.com/login.php"),
        ("PHISH", "http://shopee-khuyenmai-vip.club/claim-reward?u=9182"),
        ("PHISH", "https://secure-mbbank-update.online/dangnhap/verify"),
    ]
    s_lex, s_norms, _, s_tlds = extract_url_batch([u for _, u in stress_urls])
    s_lex["tld_phish_rate"] = [tld_rate_map.get(t, global_rate) for t in s_tlds]
    s_lex["tfidf_char_prob"] = char_clf.predict_proba(char_vec.transform(s_norms))[:, 1]
    s_lex["tfidf_word_prob"] = word_clf.predict_proba(word_vec.transform(s_norms))[:, 1]
    s_probs = lgb_model.predict_proba(s_lex[FEATURE_COLS])[:, 1]
    correct_stress = 0
    print("KIỂM THỬ ZERO-SHOT STRESS TEST (12 URL THỰC TẾ VIỆT NAM & QUỐC TẾ):")
    for (truth, u), p in zip(stress_urls, s_probs):
        pred_lbl = "PHISH" if p >= best_th else "SAFE"
        ok = "OK" if pred_lbl == truth else "SAI"
        if pred_lbl == truth:
            correct_stress += 1
        print(f"  [{ok:3s}] Thực tế={truth:5s} | AI={pred_lbl:5s} ({p*100:5.1f}%) | {u}")
    print(f"  -> Pure ML Zero-Shot Stress Test: {correct_stress}/{len(stress_urls)} ({correct_stress/len(stress_urls)*100:.1f}%)")

    bundle = {
        "name": "Full 1.1M Multi-Benchmark Phishing URL Engine (PhiUSIIL + Mitake + Pirocheto)",
        "char_vec": char_vec,
        "char_clf": char_clf,
        "word_vec": word_vec,
        "word_clf": word_clf,
        "lgb_model": lgb_model,
        "tld_rate_map": tld_rate_map,
        "global_rate": global_rate,
        "threshold": best_th,
        "feature_cols": FEATURE_COLS,
        "lexical_cols": LEXICAL_COLS,
        "feature_names_vi": FEATURE_NAMES_VI,
        "metrics": {
            "dataset": f"PhiUSIIL (UCI) + Mitake + Pirocheto ({len(df):,} URL thực tế duy nhất)",
            "split": "70% Train / 15% Val / 15% Test (Domain-Grouped — không trùng tên miền)",
            "test_size": int(len(X_test)),
            "test_phishing": int(y_test.sum()),
            "roc_auc": round(float(roc_auc), 5),
            "pr_auc": round(float(pr_auc), 5),
            "f1": round(float(f1), 5),
            "precision": round(float(prec), 5),
            "recall": round(float(rec), 5),
            "fpr": round(float(fpr), 6),
            "threshold": round(float(best_th), 4),
            "latency_ms": round(float(inf_ms), 4),
            "stress_test_accuracy": f"{correct_stress}/{len(stress_urls)}",
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "phiusiil_bundle.joblib"))


if __name__ == "__main__":
    main()
