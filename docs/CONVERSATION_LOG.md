# Lịch sử trao đổi & Quá trình nghiên cứu, huấn luyện, kiểm định AI

Tài liệu này lưu trữ toàn bộ lịch sử thảo luận từ bước phân tích đề bài **Imagine the Future Workspace (Human Centered. AI Powered)** đến bước lựa chọn dữ liệu chuẩn quốc tế, huấn luyện 3 AI Engine, kiểm định pháp y rò rỉ dữ liệu (Leakage & Bias Audit) và thử lửa trên các bộ dữ liệu độc lập bên ngoài (Cross-Dataset OOD Validation).

---

## Phần 1: Phân tích đề bài & Chốt kiến trúc AI tự chủ (Không phụ thuộc API trả phí)

### Yêu cầu đề bài
* **Chủ đề:** *Imagine the Future Workspace (Human Centered. AI Powered)* — tưởng tượng AI thay đổi cách đội ngũ fintech làm việc và cách con người mua sắm, thanh toán, quản lý tài chính, tự bảo vệ trên không gian số; biến ý tưởng thành sản phẩm chạy trực tuyến thật.
* **3 nhóm chủ đề:** Trợ lý AI, An ninh mạng, Chống lừa đảo.
* **4 đầu mục nộp bài:** Hướng dẫn sử dụng (PDF ≤ 1 trang A4 + link chạy thật), Mã nguồn (`.zip` + `README` khai báo tài sản/thư viện/AI), Slide (PDF ≤ 10 trang, không sửa sau khi nộp), Video demo chạy thật (MP4 ≤ 3 phút).

### Các câu hỏi trọng tâm & Chốt phương án
* **User:** *Này muốn test là phải có API AI đúng không? Theo mấy ý tưởng này là phải dùng API của AI hay chỉ cần code không? Hướng 1 (Fintech Fraud & Ops Workspace dùng LightGBM, Isolation Forest, Graph + SHAP không cần API) này thật không?*
* **Assistant:** Xác nhận 100% thật và là kiến trúc chuẩn ngoài thực tế tại các công ty Fintech (Stripe Radar, PayPal, ví điện tử). Không cần gọi API LLM trả phí hay lo hết quota khi BGK chấm — tự train mô hình ML/Graph chạy CPU dưới 30ms + dùng `SHAP TreeExplainer` tự động dịch các yếu tố đẩy rủi ro sang tiếng Việt tự nhiên cho nhân viên vận hành và người dùng cuối.
* **User:** *Vậy sẽ có bộ data nào để mình train? Ủa không có bộ dữ liệu thực tế à mà lại tạo ra data?*
* **Assistant:** Giải thích rõ lý do bảo mật ngân hàng (PCI-DSS, GDPR, Nghị định 13) và đề xuất tải trực tiếp các **bộ dữ liệu chuẩn công khai quốc tế (Public Benchmark Datasets)** từ Hugging Face Hub / UCI bằng Python mà không cần đăng nhập Kaggle:
  1. **PaySim** (Giao dịch ví điện tử & đồ thị tài khoản)
  2. **Sparkov Credit Card Fraud** (Thanh toán & mua sắm thẻ theo tọa độ địa lý)
  3. **PhiUSIIL Phishing URL (UCI)** (Chống link lừa đảo thực tế ngoài đời)
* **User:** *Sparkov Credit Card Fraud Dataset và PhiUSIIL Phishing URL Dataset không dùng 2 bộ này à? Kết quả của model chứ t không quan tâm train nhanh hay chậm.*

---

## Phần 2: Huấn luyện Full-Dataset cả 3 AI Engine & Phát hiện lỗi Bias trên PhiUSIIL

### Lượt huấn luyện đầu tiên (Full Dataset)
* **Module 1 (PaySim — 636.262 GD):** Đạt `ROC-AUC = 1.00000`, `PR-AUC = 1.00000`, `F1 = 0.99592`, `FP = 0`.
* **Module 2 (Sparkov — 1.048.575 GD, Chronological Split):** Lượt 1 đạt `ROC-AUC = 0.99730`, `PR-AUC = 0.91415`, `F1 = 0.84248`. Sau khi nâng cấp thêm đặc trưng nhân quả **Rolling Velocity 1h / 24h** (đếm nhịp quẹt thẻ dồn dập và tổng chi tiêu 24h quá khứ), kết quả trên **157.287 GD tương lai** nhảy vọt lên:
  * **`ROC-AUC = 0.99979` | `PR-AUC = 0.98041` | `F1 = 0.92818` | `Precision = 96.38%` | `Recall = 89.51%` | `FPR = 0.0205%`** (chỉ 32 cảnh báo nhầm / 156.334 GD thật).
* **Module 3 (PhiUSIIL URL — 235.370 URL):** Lượt 1 đạt `ROC-AUC = 0.99916` trên tập Test nội bộ, **NHƯNG** khi kiểm thử zero-shot phát hiện lỗi nghiêm trọng: `https://momo.vn` bị báo **`99.8% Lừa đảo`**.
  * **Chẩn đoán pháp y:** Trong bộ PhiUSIIL gốc, **100.00% URL sạch** đều có dạng `https://www.<domain>` (`has_www = 1.0`, `is_https = 1.0`, `has_path = 0.0`). Model đã học vẹt tiền tố `www.`!
  * **Khắc phục:** Bỏ hoàn toàn `http/https` và `www.`, bổ sung tính năng nhận diện giả mạo thương hiệu chuẩn xác (`brand_impersonation`), và hợp nhất **PhiUSIIL + Full Mitake (886k URL) + Pirocheto = 1.022.282 URL duy nhất**, huấn luyện **Dual-NLP (Char N-gram + Word/Token TF-IDF) + LightGBM** với **Domain-Grouped Split**.
  * **Kết quả trên 152.481 URL thuộc tên miền hoàn toàn mới:** **`ROC-AUC = 0.97975` | `PR-AUC = 0.97063` | `F1 = 0.91515` | `Precision = 94.85%` | `Recall = 88.40%`**. Zero-Shot Stress Test Việt Nam tăng lên **11/12 (`91.7%`)** thuần ML (`momo.vn` giảm từ `99.8%` xuống `5.6%` an toàn; 6/6 link lừa đảo bắt đúng `100%`).

---

## Phần 3: Phản biện "Model đang quá lạc quan" & Kiểm định Pháp y Rò rỉ (Leakage Audit)

* **User:** *Nguy hiểm nha, model đang quá lạc quan hay sao á.*
* **Assistant:** Đồng tình 100% và chạy ngay kiểm định Feature Importance (Gain) + Stress Audit (`training/audit_leakage.py`):
  1. **PaySim — Bắt tận tay Post-Transaction Leakage:** `97.65%` gain đến từ `is_exact_drain` (quy tắc cứng của trình mô phỏng) và các cột sau giao dịch (`newbalanceOrig`, `newbalanceDest`, `error_balance_*`).
     * **Cập nhật sạch (`training/train_paysim.py`):** Cắt bỏ hoàn toàn cột sau giao dịch, chỉ giữ 18 đặc trưng **Strict Pre-Transaction** + chia **Chronological** → Kết quả trung thực trên 95.440 GD tương lai: **`ROC-AUC = 0.99996` | `PR-AUC = 0.99019` | `F1 = 0.92720` (`Precision 97.32%`, `Recall 88.54%`)**.
  2. **Sparkov — Xác nhận sạch Leakage:** Kiểm định thêm chế độ **Cold-Start cực khó** (149.602 GD của chủ thẻ hoàn toàn mới, **không dùng lịch sử 24h**) đạt `ROC-AUC = 0.9947`, `PR-AUC = 0.8716`, `F1 = 0.7969`. Sự chênh lệch lên `F1 = 0.9282` khi có lịch sử 24h quá khứ chứng minh hiệu quả thật của Rolling Velocity, không có rò rỉ tương lai.
  3. **URL Phishing — Trung thực, không ảo:** `ROC-AUC 0.9798`, `F1 0.9152` trên 152k tên miền hoàn toàn mới.

---

## Phần 4: Kiểm thử Cross-Dataset Out-Of-Distribution (OOD) trên các bộ dữ liệu độc lập bên ngoài

* **User:** *Bạn hãy kiếm các bộ data khác dùng để làm tập test các model hiện tại.*
* **Assistant:** Tải và đánh giá trên 5 bộ dữ liệu độc lập bên ngoài (`training/external_ood_test.py` & `training/eval_cyradar_clean.py`), loại bỏ 100% URL/tài khoản trùng với tập Train:
  * **Module 3 (URL Phishing) → CyRadar Việt Nam (`huynq3Cyradar`, lọc riêng 3.065 URL thật, 0% trùng):** **`ROC-AUC = 0.9384` | `PR-AUC = 0.9351` | `F1 = 0.8613` | `Recall = 90.38%` | `Precision = 82.26%`**.
  * **Module 3 (URL Phishing) → `kmack/Phishing_urls` (4.187 URL hoàn toàn mới):** **`ROC-AUC = 0.9789` | `PR-AUC = 0.9998` | `Precision = 100.00%` | `Recall = 66.67%` | `F1 = 0.8000`**.
  * **Module 1 (PaySim Clean Pre-Tx) → `theman10/paysim` shard `0001` (500.000 GD cuối tháng, 100% tài khoản mới):** **`ROC-AUC = 1.0000` | `PR-AUC = 0.9927` | `F1 = 0.9103` | `Precision = 98.95%` | `Recall = 84.29%`**.
  * **Cross-Domain Zero-Shot (PaySim & Sparkov) → `Abdulmajeed 2022 Enterprise Bank` (5.000 GD hệ thống hoàn toàn khác):** PaySim đạt `ROC-AUC = 0.5795`, Sparkov đạt `ROC-AUC = 0.6824` — chứng minh bằng thực nghiệm hiện tượng **Domain Shift** giữa các hệ thống ngân hàng khác nhau, củng cố luận điểm kiến trúc cần **Human-in-the-Loop + Unsupervised Anomaly (Isolation Forest) + Calibration** trong Future Fintech AI Workspace.

---

## Phần 5: Đóng gói & Đẩy mã nguồn, mô hình, báo cáo kiểm định và lịch sử trao đổi lên GitHub

* **User:** *https://github.com/dothanhtuan1704-droid/MOMO.git đẩy code model test model lịch sử chat*
* **Assistant:** Khởi tạo Git repository sạch, cấu hình `.gitignore` loại trừ các file raw dataset vượt ngưỡng 100MB của GitHub, đóng gói đầy đủ 3 model (`paysim_bundle.joblib`, `sparkov_bundle.joblib`, `phiusiil_bundle.joblib`), toàn bộ code train/audit/OOD test, báo cáo kiểm định `MODEL_EVALUATION_REPORT.md` và tài liệu lịch sử trao đổi `docs/CONVERSATION_LOG.md`, sau đó push lên nhánh `main` của `https://github.com/dothanhtuan1704-droid/MOMO.git`.
