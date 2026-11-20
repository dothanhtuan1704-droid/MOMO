# Fintech Fraud & Anti-Scam AI Workspace (Core ML & OOD Evaluation)

Dự án thuộc khuôn khổ **Imagine the Future Workspace (Human Centered. AI Powered)** — tập trung giải quyết bài toán **An ninh mạng, Chống lừa đảo và Trợ lý AI vận hành tài chính** bằng hệ thống **3 AI Engine tự chủ hoàn toàn (Self-hosted CPU, không phụ thuộc API AI trả phí)** kèm lớp giải thích **Explainable AI (SHAP)** và quy trình kiểm định chống rò rỉ dữ liệu & kiểm thử chéo đa bộ dữ liệu (**Cross-Dataset OOD Validation**).

---

## 1. Cấu trúc kho mã nguồn

```text
MOMO/
├── models/                         # Các mô hình AI đã huấn luyện sẵn (tổng ~40 MB, chạy CPU < 10ms)
│   ├── paysim_bundle.joblib        # Module 1: PaySim Clean Pre-Transaction + Isolation Forest + Graph
│   ├── sparkov_bundle.joblib       # Module 2: Sparkov Payment & Shopping Fraud (Velocity + Geo)
│   └── phiusiil_bundle.joblib      # Module 3: 1.02M Multi-Benchmark De-Biased Phishing URL Engine
├── training/                       # Pipeline huấn luyện, kiểm định rò rỉ & kiểm thử OOD bên ngoài
│   ├── train_paysim.py             # Huấn luyện Module 1 (Strict Pre-Transaction, Chronological)
│   ├── train_sparkov.py            # Huấn luyện Module 2 (Chronological + Rolling Velocity 1h/24h)
│   ├── train_phiusiil.py           # Huấn luyện Module 3 (1.02M URL, Dual-NLP TF-IDF + LightGBM)
│   ├── audit_leakage.py            # Kiểm định pháp y rò rỉ dữ liệu (Post-Tx Leakage & Cold-Start)
│   ├── external_ood_test.py        # Kiểm thử Cross-Dataset OOD trên 5 bộ dữ liệu độc lập bên ngoài
│   └── eval_cyradar_clean.py       # Kiểm định chuyên sâu trên URL thực tế của CyRadar Việt Nam
├── data/                           # Mẫu dữ liệu kiểm thử nhẹ (file raw lớn tự động tải từ Hugging Face)
│   ├── paysim_ops_samples.parquet  # 200 giao dịch thực tế PaySim (từ tập Test tương lai)
│   └── sparkov_ops_samples.parquet # 200 giao dịch thực tế Sparkov (từ tập Test tương lai)
├── docs/
│   └── CONVERSATION_LOG.md         # Toàn bộ lịch sử trao đổi, phân tích, thiết kế & kiểm định
├── MODEL_EVALUATION_REPORT.md      # Báo cáo kiểm định chi tiết (In-Domain, Audit Leakage & External OOD)
└── requirements.txt
```

---

## 2. Tóm tắt kết quả 3 AI Engine (Đánh giá trung thực — Zero Data Leakage)

Xem chi tiết toàn bộ bảng đối chiếu và phân tích tại [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md).

| Module AI | Nguồn dữ liệu huấn luyện | Kiểm định nội bộ (Unseen Future / Domain) | Kiểm định bên ngoài (External OOD) |
| :--- | :--- | :--- | :--- |
| **1. Chuyển khoản & Mạng lưới** | **PaySim** (636.262 GD, Strict Pre-Tx) | `ROC-AUC 0.99996` \| `PR-AUC 0.99019` \| `F1 0.9272` | **theman10/paysim (500k tài khoản mới):** `ROC-AUC 1.0000`, `PR-AUC 0.9927`, `F1 0.9103` |
| **2. Thanh toán & Mua sắm Thẻ** | **Sparkov** (1.048.575 GD, Chronological) | `ROC-AUC 0.99979` \| `PR-AUC 0.98041` \| `F1 0.9282` | **Cold-Start chủ thẻ mới (149k GD):** `ROC-AUC 0.9947`, `F1 0.7969` |
| **3. Chống Link Lừa đảo (URL)** | **PhiUSIIL (UCI) + Mitake + Pirocheto** (1.022.282 URL duy nhất, đã khử bias `www.`) | `ROC-AUC 0.97975` \| `PR-AUC 0.97063` \| `F1 0.9152` | **CyRadar VN (3.065 URL mới):** `ROC-AUC 0.9384`, `Recall 90.38%`, `F1 0.8613`<br>**kmack (4.187 URL mới):** `ROC-AUC 0.9789`, `Precision 100%` |

---

## 3. Hướng dẫn cài đặt & tái lập (Reproducibility)

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy huấn luyện lại từ đầu (tự động tải dataset chuẩn từ Hugging Face Hub)
```bash
python training/train_paysim.py
python training/train_sparkov.py
python training/train_phiusiil.py
```

### Chạy kiểm định pháp y rò rỉ & kiểm thử Cross-Dataset OOD bên ngoài
```bash
python training/audit_leakage.py
python training/external_ood_test.py
python training/eval_cyradar_clean.py
```

---

## 4. Khai báo minh bạch Tài sản có sẵn, Dữ liệu & Thư viện mã nguồn mở

### Bộ dữ liệu công khai (Public Benchmark Datasets)
1. **PaySim** (`flwrlabs/fed-fraud-paysim-banks` & `theman10/paysim` trên Hugging Face Hub) — Lopez-Rojas et al., Mobile Money Financial Transactions Benchmark.
2. **Sparkov Credit Card Fraud** (`santosh3110/credit_card_fraud_transactions` trên Hugging Face Hub) — 1.048.575 giao dịch thẻ tín dụng có tọa độ địa lý, thời gian và danh mục mua sắm.
3. **PhiUSIIL Phishing URL Dataset** (`it4lia/PhiUSIIL_cleaned` — UCI Machine Learning Repository), kết hợp **Mitake Phishing & Benign URLs** (`Mitake/PhishingURLsANDBenignURLs`) và **Pirocheto Phishing URL** (`pirocheto/phishing-url`).
4. **Bộ dữ liệu kiểm thử độc lập bên ngoài (External OOD Test Sets):**
   - **CyRadar Vietnam Phishing Detection Dataset** (`huynq3Cyradar/Phishing_Detection_Dataset`).
   - **kmack Phishing URLs** (`kmack/Phishing_urls`).
   - **Abdulmajeed 2022 Enterprise Bank Transactions** (`Abdulmajeedyahya/1M-Fraud-Labeled-Synthetic-Bank-Transactions`) — dùng để đo lường thực tế hiện tượng *Domain Shift* xuyên hệ thống ngân hàng.

### Thư viện mã nguồn mở & Công cụ AI hỗ trợ phát triển
* **ML / Graph / Explainability:** `LightGBM`, `scikit-learn` (`IsolationForest`, `TfidfVectorizer`, `LogisticRegression`), `SHAP` (`TreeExplainer`), `NetworkX`, `pandas`, `numpy`, `pyarrow`, `joblib`.
* **Công cụ AI hỗ trợ lập trình & kiểm định:** Google Antigravity AI Assistant.
