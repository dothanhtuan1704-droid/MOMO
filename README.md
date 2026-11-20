# Fintech Fraud & Anti-Scam AI Workspace (Real-World Core ML & Live Verification)

Dự án thuộc khuôn khổ **Imagine the Future Workspace (Human Centered. AI Powered)** — giải quyết bài toán **An ninh mạng, Chống lừa đảo và Trợ lý AI vận hành tài chính** bằng hệ thống **AI tự chủ hoàn toàn (Self-hosted CPU, không phụ thuộc API AI trả phí)** kèm lớp giải thích **Explainable AI (SHAP tiếng Việt)**.

Toàn bộ các mô hình lõi đều được huấn luyện và kiểm định khắt khe trên **100% DỮ LIỆU THẬT NGOÀI ĐỜI (0% Synthetic)** bao gồm **590.540 giao dịch thương mại điện tử thật của Vesta Corporation (IEEE-CIS)** và **1,02 triệu URL thật** được thử lửa trực tiếp trên các nguồn tấn công đang hoạt động sống ngoài Internet (**OpenPhish LIVE, Phishing.Database ACTIVE, CERT.pl LIVE, URLhaus LIVE, CyRadar Việt Nam, HackerNews & Báo chí VN RSS LIVE**).

---

## 1. Cấu trúc kho mã nguồn

```text
MOMO/
├── models/                             # Các mô hình AI đã huấn luyện sẵn (chạy CPU < 50ms)
│   ├── vesta_real_bundle.joblib        # LÕI CHÍNH 1: IEEE-CIS Vesta Real-World Payment Fraud (590k GD thật)
│   ├── phiusiil_bundle.joblib          # LÕI CHÍNH 2: 1.02M Real-World De-Biased Anti-Phishing URL Engine
│   ├── paysim_bundle.joblib            # Module bổ trợ: PaySim Clean Pre-Tx + NetworkX Graph Explorer
│   └── sparkov_bundle.joblib           # Baseline giả lập (giữ lại làm đối chứng Domain Shift trên dữ liệu thật)
├── training/                           # Pipeline huấn luyện, kiểm định rò rỉ & thử lửa dữ liệu sống
│   ├── train_vesta_real.py             # Huấn luyện Lõi 1 trên 590,540 GD thật IEEE-CIS Vesta (OOF + Velocity)
│   ├── train_phiusiil.py               # Huấn luyện Lõi 2 trên 1.02M URL thật (Dual-NLP TF-IDF + LightGBM)
│   ├── train_paysim.py                 # Huấn luyện PaySim Clean Pre-Transaction + NetworkX Graph
│   ├── train_sparkov.py                # Huấn luyện baseline Sparkov (để đối chứng với dữ liệu thật)
│   ├── test_real_world_data.py         # Thử lửa 0% synthetic: OpenPhish, Phishing.Database, OpenDNS & IEEE-CIS Vesta
│   ├── verify_real_models.py           # Stress test sâu: Tiền $USD cứu được, Cold-Start thẻ mới, 20 ca VN & SHAP
│   ├── test_extra_real_datasets.py     # Thử lửa mở rộng: CERT.pl, URLhaus, HackerNews, 208 bài báo VN & Ép lỗi Vesta
│   ├── external_ood_test.py            # Kiểm thử Cross-Dataset OOD (theman10/paysim, kmack, CyRadar, Abdulmajeed)
│   ├── eval_cyradar_clean.py           # Kiểm định riêng trên URL thật của CyRadar Việt Nam
│   └── audit_leakage.py                # Kiểm định pháp y rò rỉ dữ liệu (Post-Tx Leakage Audit)
├── data/                               # Mẫu nhẹ cho UI (file raw >100MB tự động tải từ Hugging Face / Live Feeds)
│   ├── vesta_real_ops_samples.parquet  # 200 giao dịch thật IEEE-CIS Vesta (từ tập Test tương lai)
│   ├── paysim_ops_samples.parquet      # 200 giao dịch PaySim Clean Pre-Tx
│   └── sparkov_ops_samples.parquet     # 200 giao dịch Sparkov
├── docs/
│   └── CONVERSATION_LOG.md             # Toàn bộ lịch sử trao đổi, phản biện, audit & kiểm định thực tế
├── MODEL_EVALUATION_REPORT.md          # Báo cáo kiểm định chi tiết (Real-World Live & Robustness Stress Tests)
└── requirements.txt
```

---

## 2. Tóm tắt kết quả trên 100% DỮ LIỆU THẬT NGOÀI ĐỜI (0% Synthetic)

Xem đầy đủ bảng số liệu, các ca báo nhầm thực tế và kết quả bài ép lỗi tại [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md).

| Lõi AI chính | Dữ liệu huấn luyện (Thật 100%) | Kết quả kiểm định thực tế (Zero Leakage / Live Attacks) | Tốc độ & Tác động |
| :--- | :--- | :--- | :--- |
| **1. Thanh toán TMĐT THẬT** (`vesta_real_bundle.joblib`) | **IEEE-CIS Vesta Corp**<br>590.540 GD thật (`20.663` gian lận thật) | • **88.581 GD tương lai thật:** `ROC-AUC 0.8921` \| `PR-AUC 0.4798` (**Lift 13.8x**)<br>• **4 tuần tương lai liên tiếp:** `AUC 0.880 – 0.905` (không tụt phong độ)<br>• **Cold-Start đầu thẻ mới:** `ROC-AUC 0.9454` \| `Unseen UID`: `0.8678`<br>• **Xóa trắng 10 cột V-block:** vẫn giữ `ROC-AUC 0.8844` | `49 ms` (gồm SHAP)<br>Cứu **\$150,075** @ `1% FPR` và **\$240,659** @ `5% FPR` |
| **2. Chống Link Lừa đảo THẬT** (`phiusiil_bundle.joblib`) | **PhiUSIIL (UCI) + Mitake + Pirocheto**<br>1.022.282 URL thật duy nhất (đã khử bias `www.`) | • **OpenPhish LIVE hôm nay (300 link):** Bắt **`91.33%`** (nhóm tinh vi `HTTPS+.com/.org`: `70.6%`)<br>• **Phishing.Database ACTIVE (10k link):** Bắt **`92.90%`**<br>• **CyRadar Việt Nam (3.065 URL mới):** `ROC-AUC 0.9384` \| `Recall 90.38%`<br>• **208 link báo VN LIVE (`VNExpress/TuoiTre/DanTri`) & 20 web VN thật:** An toàn **`100.00%`** (`0` báo nhầm)<br>• **HackerNews LIVE (87 URL sâu):** An toàn **`95.40%`**<br>• **Hard Live Test (20k CERT.pl+URLhaus vs 295 HN+Báo VN):** `ROC-AUC 0.9477` | `0.55 ms / URL`<br>**20/20 (`100%`)** kịch bản thật VN thuần ML |

---

## 3. Hướng dẫn cài đặt & tái lập kiểm định thực tế

```bash
pip install -r requirements.txt

# Huấn luyện 2 lõi AI 100% dữ liệu thật
python training/train_vesta_real.py
python training/train_phiusiil.py

# Chạy toàn bộ bộ kiểm thử dữ liệu sống thực tế (Live Feeds & Stress Tests)
python training/test_real_world_data.py
python training/verify_real_models.py
python training/test_extra_real_datasets.py
```

---

## 4. Khai báo minh bạch Dữ liệu công khai, Thư viện & Công cụ AI

* **Dữ liệu giao dịch thật (0% Synthetic):** IEEE-CIS Fraud Detection — Vesta Corporation (`t8nisho/vesta-fraud-data` trên Hugging Face).
* **Dữ liệu URL thật & Threat Intel LIVE:** PhiUSIIL (UCI), Mitake Phishing & Benign URLs, Pirocheto Phishing URL, CyRadar Vietnam (`huynq3Cyradar/Phishing_Detection_Dataset`), `kmack/Phishing_urls`, OpenPhish Public Feed, Phishing.Database ACTIVE, CERT.pl National Cybersecurity Center, URLhaus (abuse.ch), OpenDNS Top Real Domains, HackerNews API, RSS VNExpress / Tuổi Trẻ / Dân Trí.
* **Dữ liệu mô phỏng (chỉ dùng cho Graph Explorer & đối chứng Domain Shift):** PaySim (`flwrlabs/fed-fraud-paysim-banks`, `theman10/paysim`), Sparkov (`santosh3110/credit_card_fraud_transactions`), Abdulmajeed 2022.
* **Thư viện mã nguồn mở:** `LightGBM`, `scikit-learn`, `SHAP`, `NetworkX`, `pandas`, `numpy`, `pyarrow`, `joblib`.
* **Công cụ AI hỗ trợ phát triển & kiểm định:** Google Antigravity AI Assistant.
