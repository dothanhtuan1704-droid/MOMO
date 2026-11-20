# Báo cáo Kiểm định Mô hình AI — In-Distribution, Leakage Audit & Cross-Dataset OOD Validation

Tài liệu này ghi nhận toàn bộ kết quả thực nghiệm, các phát hiện pháp y dữ liệu (Leakage & Collection Bias Audit) và kết quả thử lửa trên các bộ dữ liệu độc lập bên ngoài (External Out-Of-Distribution Testing) của 3 AI Engine.

---

## 1. Module 1 — Chuyển khoản & Mạng lưới Tài khoản (PaySim)

### 1.1. Phát hiện & Xử lý Post-Transaction Leakage (Audit Pháp y)
* **Phiên bản thử nghiệm ban đầu** đạt `ROC-AUC = 1.00000`, `PR-AUC = 1.00000`, `F1 = 0.99592` (`FP = 0`).
* **Kiểm định Feature Importance (Gain)** phát hiện `99.97%` sức mạnh đến từ các đặc trưng rò rỉ sau giao dịch (*Post-Transaction Leakage*) và quy tắc cứng của bộ mô phỏng:
  * `is_exact_drain` (`amount == oldbalanceOrg` chính xác từng đồng lẻ): `97.65%` gain.
  * `orig_balance_diff`, `dest_unchanged_large`, `newbalanceOrig`, `error_balance_orig`, `error_balance_dest`: dùng số dư **sau khi giao dịch đã hoàn tất** (không có thật ở thời điểm cần chặn trước khi chuyển tiền).
* **Khắc phục triệt để (`training/train_paysim.py`):**
  1. Loại bỏ hoàn toàn `newbalanceOrig`, `newbalanceDest`, `error_balance_orig`, `error_balance_dest`, `is_exact_drain`.
  2. Chỉ dùng **18 đặc trưng Pre-Transaction** (biết trước khi phê duyệt), làm tròn tỷ lệ 2 chữ số thập phân, tính tần suất đồ thị **nhân quả chỉ nhìn quá khứ (`cumcount`)** và chia theo **thời gian thực (`Chronological 70/15/15`)**.

### 1.2. Kết quả kiểm định chính thức (`models/paysim_bundle.joblib`)
| Kịch bản kiểm thử | Số mẫu | Ca gian lận | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chronological Test tương lai** (`paysim_raw`, 15% cuối) | 95.440 | 410 | **0.99996** | **0.99019** | **0.92720** | 97.32% | 88.54% | 0.0105% |
| **External Test 1A:** `theman10/paysim` shard `0001` (100% tài khoản mới, cuối tháng) | 500.000 | 3.915 | **1.0000** | **0.9927** | **0.9103** | 98.95% | 84.29% | 0.01% |
| **External Cross-Domain 1B:** `Abdulmajeed 2022 Bank` *(hệ thống ngân hàng hoàn toàn khác)* | 5.000 | 73 | 0.5795 | 0.0176 | 0.0000 | 0.00% | 0.00% | 0.00% |

---

## 2. Module 2 — Thanh toán & Mua sắm Thẻ (Sparkov Credit Card Fraud)

### 2.1. Thiết kế đặc trưng nhân quả (Zero Leakage)
* Tập dữ liệu gốc: **1.048.575 giao dịch** (`6.006` ca gian lận, `0.57%`), sắp xếp theo `unix_time` và chia **Chronological 70% Train / 15% Val / 15% Test**.
* Toàn bộ đặc trưng hành vi (`time_since_last_tx_sec`, `travel_speed_kmh` — **Impossible Travel**, `tx_count_last_1h`, `tx_count_last_24h`, `amt_sum_last_24h`, `amt_ratio_user_mean`) chỉ sử dụng lịch sử quá khứ nghiêm ngặt (`shift >= 1`).

### 2.2. Kết quả kiểm định chính thức (`models/sparkov_bundle.joblib`)
| Kịch bản kiểm thử | Số mẫu | Ca gian lận | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold-Start Audit:** Chủ thẻ hoàn toàn mới (`GroupSplit`), **không dùng lịch sử 24h** | 149.602 | 918 | **0.9947** | **0.8716** | **0.7969** | 87.90% | 72.90% | — |
| **Production (Lượt 1 — chưa có Rolling 1h/24h)** — Future Test | 157.287 | 953 | 0.99730 | 0.91415 | 0.84248 | 96.23% | 74.92% | 0.0179% |
| **Production (Bản cuối — có Rolling Velocity 1h/24h)** — Future Test | 157.287 | 953 | **0.99979** | **0.98041** | **0.92818** | **96.38%** | **89.51%** | **0.0205%** |
| **External Cross-Domain 2A:** `Abdulmajeed 2022 Bank/Card` *(khác hẳn hệ thống)* | 5.000 | 73 | 0.6824 | 0.0275 | 0.0631 | 3.83% | 17.81% | 6.62% |

* **Nhận xét kỹ thuật:** Bổ sung Rolling Velocity 1h/24h hợp lệ quá khứ giúp tăng PR-AUC từ `0.914` → **`0.9804`** và Recall từ `74.9%` → **`89.5%`** (chỉ 32 cảnh báo nhầm / 156.334 giao dịch thật).

---

## 3. Module 3 — Lá chắn Chống Link Lừa đảo (Multi-Benchmark URL Phishing Engine)

### 3.1. Phát hiện & Khử Collection Bias kinh điển trên bộ PhiUSIIL (UCI)
* Khi kiểm định zero-shot bằng URL thực tế Việt Nam (`https://momo.vn`), phát hiện **100.00% URL an toàn trong bộ PhiUSIIL UCI gốc** đều có dạng `https://www.<domain>` và `path_len == 0` — khiến model học vẹt tiền tố `www.` và đánh nhầm `https://momo.vn` thành `99.8%` lừa đảo.
* **Giải pháp xử lý triệt để (`training/train_phiusiil.py`):**
  1. Chuẩn hóa bỏ hoàn toàn scheme (`http/https`) và tiền tố `www.` trước khi trích xuất đặc trưng & TF-IDF.
  2. Hợp nhất **3 nguồn dữ liệu thực tế đa đường dẫn**: **PhiUSIIL (UCI)** + **Full Mitake (886k URL)** + **Pirocheto** → **1.022.282 URL duy nhất**.
  3. Chia tập bằng **Domain-Grouped Split** (100% tên miền tập Test chưa từng xuất hiện trong Train).
  4. Kiến trúc 2 tầng: **Dual-NLP Out-Of-Fold (`Char N-gram (3,5)` + `Word/Token TF-IDF`)** kết hợp **LightGBM 30 đặc trưng** (bao gồm `brand_impersonation`, `domain_entropy`, `vowel_ratio_domain`, `tld_phish_rate`).

### 3.2. Kết quả kiểm định nội bộ & Cross-Dataset OOD (`models/phiusiil_bundle.joblib`)
| Kịch bản kiểm thử (0% trùng URL với tập Train) | Số URL | Phishing | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Domain-Grouped Test** (100% tên miền chưa từng thấy) | 152.481 | 49.837 | **0.97975** | **0.97063** | **0.91515** | 94.85% | 88.40% | 2.33% |
| **External OOD 3A:** `kmack/Phishing_urls` *(đã loại 94.1% URL trùng)* | 4.187 | 4.155 | **0.9789** | **0.9998** | **0.8000** | **100.00%** | 66.67% | **0.00%** |
| **External OOD 3C:** **CyRadar Việt Nam** *(lọc riêng URL thật, 0% trùng)* | 3.065 | 1.216 | **0.9384** | **0.9351** | **0.8613** | 82.26% | **90.38%** | 12.82% |

### 3.3. Kết quả Zero-Shot Stress Test 12 URL Việt Nam & Quốc tế (Thuần ML, không dùng Whitelist)
* **Độ chính xác:** **11/12 (`91.7%`)** — bắt trúng **6/6 (`100%`)** link lừa đảo giả mạo (`momo-nhanqua-tri-an-2026.xyz`, `vietcombank-xacminh-baomat.top`, `192.168.45.12/...`, `paypal-verify-account.firebaseapp.com`, `shopee-khuyenmai-vip.club`, `secure-mbbank-update.online`) và nhận đúng `momo.vn` (`5.6%`), `vietcombank.com.vn/ca-nhan` (`14.3%`), `techcombank.com` (`0.5%`), `google.com` (`2.0%`), `vnexpress.net` (`12.5%`).

---

## 4. Kết luận kiến trúc từ bài thử nghiệm Domain Shift
* **URL Phishing (Module 3)** có tính tổng quát toàn cầu cao (`ROC-AUC 0.9384 – 0.9789` trên dữ liệu độc lập hoàn toàn như CyRadar VN và kmack) vì cấu trúc HTTP/DNS là chuẩn chung.
* **Transaction Fraud (Module 1 & 2)** chịu tác động mạnh của **Domain Shift** khi áp zero-shot từ hệ thống ngân hàng A sang hệ thống ngân hàng B (`AUC` tụt từ `0.99` xuống `0.58 – 0.68` trên bộ Abdulmajeed 2022). Điều này khẳng định thiết kế **Human-in-the-Loop + Unsupervised Isolation Forest + Calibration theo từng tổ chức** là bắt buộc trong một **Future Fintech AI Workspace** thực tế.
