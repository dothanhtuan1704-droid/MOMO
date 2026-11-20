# Báo cáo Kiểm định Mô hình AI — 100% Dữ liệu Thật Ngoài Đời, Audit Rò rỉ & Stress Tests

Tài liệu này ghi nhận toàn bộ kết quả thực nghiệm minh bạch (bao gồm cả điểm mạnh, ca báo nhầm và giới hạn thực tế) sau quá trình kiểm định gắt gao trên **dữ liệu thật 100% (0% Synthetic)** và các nguồn tấn công đang hoạt động sống ngoài Internet.

---

## 1. Phát hiện bước ngoặt: Tại sao không được tin mù quáng vào Data Giả lập (Synthetic)?

Trong quá trình kiểm thử phản biện, đội ngũ đã mang mô hình **Sparkov** (vốn đạt `ROC-AUC = 0.99979` trên tập test giả lập) đi chấm zero-shot trên **590.540 giao dịch thanh toán thương mại điện tử THẬT của người thật (IEEE-CIS Vesta Corporation)**:
* **Kết quả thực tế:** `ROC-AUC` tụt từ `0.9998` xuống **`0.4828`** (`Recall = 1.31%`, thua cả đoán mò!).
* **Kết luận:** Dữ liệu giả lập tạo ra quy luật quá nhân tạo. Vì vậy, đội đã chuyển hẳn lõi phát hiện gian lận thanh toán sang huấn luyện trực tiếp trên **590.540 giao dịch thật của IEEE-CIS Vesta (0% Synthetic)** kết hợp cùng lõi **1,02 triệu URL thật**.

---

## 2. Lõi AI 1 — Thanh toán TMĐT THẬT (`models/vesta_real_bundle.joblib`)

* **Dữ liệu:** **590.540 giao dịch thật** của Vesta Corporation (`20.663` ca gian lận thật = `3.50%`), chia nghiêm ngặt theo thời gian thực **Chronological 70% Train (`413.378`) / 15% Val (`88.581`) / 15% Future Test (`88.581`, gồm `3.083` gian lận thật)**.
* **Kỹ thuật chống Overfit:** 5-Fold Out-Of-Fold Smoothed Target Encoding + Rolling Velocity 1h/24h nhân quả quá khứ + 70 tín hiệu thanh toán thật + Isolation Forest Zero-Day Anomaly.

### 2.1. Hiệu quả tài chính thực tế trên 88.581 giao dịch tương lai thật (`$469,608.52` tổng thiệt hại)
| Chế độ vận hành (Human-in-the-Loop) | FPR | Recall (Bắt đúng) | Tiền gian lận cứu được (`$USD`) | Tỷ lệ tiền cứu được |
| :--- | :---: | :---: | :---: | :---: |
| **Tầng Đỏ — Chặn tự động siêu chặt** | `1.00%` | `42.17%` | **\$150,075.00** | **32.0%** |
| **Ngưỡng tối ưu F1** (`th = 0.1809`) | `2.02%` | `48.98%` | **\$185,491.14** | **39.5%** |
| **Tầng Cam — Bàn duyệt Fraud Ops** | `3.00%` | `52.81%` | — | — |
| **Tầng Vàng — Step-up OTP / Cảnh báo** | `5.00%` | `58.19%` | **\$240,659.76** | **51.2%** |

* **Chỉ số tổng hợp tập Test tương lai:** `ROC-AUC = 0.89207` | `PR-AUC = 0.47980` (**Lift 13.8x** so với baseline `0.0348`) | `F1 = 0.47785` | `Precision = 46.65%` | `Recall = 48.98%` | Độ trễ Live Inference + SHAP: **`49.0 ms`**.

### 2.2. Các bài Stress Test khắc nghiệt trên Vesta thật
* **Cold-Start (Chủ thẻ & Đầu thẻ hoàn toàn mới chưa từng có trong Train):**
  * `Unseen UID` (`4.984` GD, `187` gian lận): **`ROC-AUC = 0.8678`** | `PR-AUC = 0.4467` | `Recall = 43.3%`.
  * `Unseen card1` (`1.210` GD, `46` gian lận): **`ROC-AUC = 0.9454`** | `PR-AUC = 0.5749` | `Recall = 60.9%`.
* **Ổn định theo thiết bị thật:**
  * `Mobile` (`7.721` GD): **`ROC-AUC = 0.9104`** | `PR-AUC = 0.6552`.
  * `Desktop` (`10.461` GD): **`ROC-AUC = 0.8947`** | `PR-AUC = 0.6771`.
* **Độ trôi theo thời gian (4 tuần tương lai liên tiếp — KHÔNG tụt phong độ):**
  * Tuần 1: `AUC 0.8964` (`PR-AUC 0.5421`) → Tuần 2: `AUC 0.8842` → Tuần 3: `AUC 0.8800` → Tuần 4 *(xa nhất)*: **`AUC 0.9051`** (`PR-AUC 0.5372`, `F1 0.5346`).
* **Bài ép lỗi che dữ liệu (Feature Masking Robustness):**
  * Đủ 70 cột: `ROC-AUC = 0.8921`.
  * **Xóa trắng 100% 10 cột V-block (`vesta_v_* = NaN`):** vẫn giữ **`ROC-AUC = 0.8844`** *(chỉ giảm `0.0077`)*.
  * **Xóa trắng 100% toàn bộ lịch sử OOF (`card1/card2/addr1/email/device`):** vẫn giữ **`ROC-AUC = 0.8513`**.

---

## 3. Lõi AI 2 — Chống Link Lừa đảo THẬT (`models/phiusiil_bundle.joblib`)

* **Xử lý Collection Bias PhiUSIIL:** Phát hiện `100%` link sạch PhiUSIIL gốc đều có `https://www.` và không có path (gây báo nhầm `momo.vn` thành `99.8%` lừa đảo). Đã chuẩn hóa loại bỏ `http/https` và `www.`, hợp nhất **PhiUSIIL + Full Mitake + Pirocheto = 1.022.282 URL thật duy nhất**, huấn luyện **Dual-NLP Out-Of-Fold (`Char N-gram (3,5)` + `Word/Token TF-IDF`) + LightGBM 30 đặc trưng** chia theo **Domain-Grouped Split**.

### 3.1. Bảng kết quả toàn diện trên các nguồn Tấn công Sống & Website Thật (0% trùng Train)
| Nguồn kiểm thử độc lập (100% dữ liệu thật) | Số mẫu mới | Chỉ số chính | Ghi chú trung thực |
| :--- | :---: | :--- | :--- |
| **Domain-Grouped Test** (tên miền hoàn toàn mới) | 152.481 URL | `ROC-AUC 0.9798` \| `PR-AUC 0.9706` \| `F1 0.9152` | `Prec 94.85%`, `Rec 88.40%`, `FPR 2.33%` |
| **OpenPhish LIVE hôm nay** (tấn công đang sống) | 300 URL | **Recall = `91.33%`** (`274/300`) | Riêng nhóm siêu tinh vi `HTTPS+.com/.org/.net`: **`70.59%`** (`24/34`) |
| **Phishing.Database ACTIVE hôm nay** | 10.000 URL | **Recall = `92.90%`** (`9.290/10.000`) | Lấy mẫu ngẫu nhiên từ `789.052` link đang sống |
| **LIVE Benchmark** (OpenPhish+PhishingDB vs OpenDNS) | 16.749 URL | **`ROC-AUC 0.9623`** \| **`F1 0.9286`** | `Prec 92.86%`, `Rec 92.85%`, `FPR 11.4%` (CDN ngắn) |
| **CyRadar Việt Nam** (lọc riêng URL thật) | 3.065 URL | **`ROC-AUC 0.9384`** \| **`F1 0.8613`** | `Recall 90.38%`, `Prec 82.26%` |
| **kmack/Phishing_urls** (loại 94.1% trùng) | 4.187 URL | **`ROC-AUC 0.9789`** \| **`Prec 100.00%`** | `FP = 0`, `Recall 66.67%` |
| **Báo chí VN LIVE RSS hôm nay** (`VNExpress/TuoiTre/DanTri`) | 208 URL sâu | **An toàn = `100.00%`** (`208/208`) | **0 báo nhầm** dù URL dài có nhiều `-`, số ID, `.html` |
| **20 kịch bản thực tế VN** (10 web thật vs 10 chiêu lừa đảo) | 20 URL | **Chính xác = `100.00%`** (`20/20`) | Thuần ML (`0.55 ms/URL`), không dùng Whitelist |
| **HackerNews LIVE hôm nay** (URL bài viết sâu quốc tế) | 87 URL sâu | **An toàn = `95.40%`** (`83/87`) | 4 ca báo nhầm: 3 ca `*.github.io` + 1 ca Mastodon chứa `@` |
| **CERT.pl LIVE hôm nay** *(chỉ có tên miền trần, thiếu path)* | 10.000 domain | **Recall = `67.35%`** (`6.735/10.000`) | Thiếu đường dẫn phụ `/login` nên khó hơn URL đầy đủ |
| **URLhaus (abuse.ch) LIVE** *(máy chủ bị hack chứa malware)* | 10.000 URL | **Recall = `50.97%`** (`5.097/10.000`) | Website sạch bị hack chèn malware khó nhận diện thuần qua chuỗi URL |
| **Tổng hợp Hard Live Test** (`20k` CERT.pl+URLhaus vs `295` HN+Báo VN) | 20.295 URL | **`ROC-AUC = 0.9477`** | Khả năng phân tách thực tế vững vàng |

---

## 4. Audit bổ sung trên Module Graph (PaySim Clean Pre-Transaction)
* Đã loại bỏ hoàn toàn Post-Transaction Leakage (`newbalanceOrig`, `newbalanceDest`, `error_balance_*`, `is_exact_drain`), chỉ giữ 18 đặc trưng **Strict Pre-Transaction** + chia **Chronological**:
  * **Chronological Future Test (`95.440` GD):** `ROC-AUC = 0.99996` | `PR-AUC = 0.99019` | `F1 = 0.92720`.
  * **External `theman10/paysim` shard `0001` (`500.000` tài khoản mới):** `ROC-AUC = 1.0000` | `PR-AUC = 0.9927` | `F1 = 0.9103`.
  * **Cross-Domain sang hệ thống khác (`Abdulmajeed 2022`):** `ROC-AUC = 0.5795` (xác nhận Domain Shift giữa các hệ thống ngân hàng khác luật).
