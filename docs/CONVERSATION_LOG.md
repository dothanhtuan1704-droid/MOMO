# Lịch sử trao đổi & Quá trình nghiên cứu, huấn luyện, kiểm định AI

Tài liệu này lưu trữ toàn bộ lịch sử thảo luận từ bước phân tích đề bài **Imagine the Future Workspace (Human Centered. AI Powered)** đến bước phát hiện bẫy dữ liệu giả lập (synthetic), chuyển sang huấn luyện & thử lửa trên **100% dữ liệu thật ngoài đời (IEEE-CIS Vesta & Live Phishing Feeds hôm nay)**.

---

## Phần 1: Phân tích đề bài & Chốt kiến trúc AI tự chủ (Không phụ thuộc API trả phí)
* **Chủ đề:** *Imagine the Future Workspace (Human Centered. AI Powered)* — phủ 3 mảng Trợ lý AI, An ninh mạng, Chống lừa đảo.
* **User:** Hỏi rõ liệu có bắt buộc gọi API LLM trả phí không, và kiểm chứng khả năng tự train mô hình ML + Graph + SHAP chạy trực tiếp trên CPU không cần API.
* **Assistant:** Xác nhận kiến trúc chuẩn Fintech thực tế (`LightGBM + Isolation Forest + NetworkX Graph + SHAP TreeExplainer`) chạy CPU `< 50ms`, không lo hết quota API, và đề xuất tải các bộ dữ liệu chuẩn công khai (PaySim, Sparkov, PhiUSIIL).

## Phần 2: Huấn luyện ban đầu & Audit pháp y phát hiện Bias / Leakage
* **User:** Yêu cầu dùng đủ cả 3 bộ dữ liệu và nhấn mạnh chỉ quan tâm **chất lượng thật của model**, sau đó phản biện thẳng: *"Nguy hiểm nha, model đang quá lạc quan hay sao á"*.
* **Phát hiện 1 (PhiUSIIL Bias):** 100% URL sạch trong PhiUSIIL gốc có `https://www.` và không có path, làm model đánh nhầm `https://momo.vn` thành `99.8%` lừa đảo. Đã khử hoàn toàn `www.`/scheme, hợp nhất **PhiUSIIL + Full Mitake + Pirocheto = 1.022.282 URL thật duy nhất**, huấn luyện Dual-NLP TF-IDF + LightGBM (Domain-Grouped Split) → `ROC-AUC 0.9798`, `F1 0.9152`.
* **Phát hiện 2 (PaySim Post-Tx Leakage):** Audit Feature Importance bóc trần `97.65%` gain đến từ `is_exact_drain` và cột sau giao dịch (`newbalanceOrig/Dest`). Đã viết lại bản **Strict Pre-Transaction + Chronological** (`training/train_paysim.py`) → `F1 = 0.9272`, kiểm chứng trên `500.000` tài khoản mới `theman10/paysim` đạt `F1 = 0.9103`.
* **Phát hiện 3 (External OOD Test):** Chạy thử lửa trên **CyRadar Việt Nam** (`3.065` URL thật mới, 0% trùng) đạt **`ROC-AUC = 0.9384`, `Recall = 90.38%`**; trên **`kmack`** (`4.187` URL mới) đạt **`ROC-AUC = 0.9789`, `Precision = 100%`**.

## Phần 3: Phản biện "Test bằng data thực tế, không dùng data gen!" & Bước ngoặt IEEE-CIS Vesta
* **User:** *"Bạn hãy test bằng data thực tế chứ không phải là gen data để test :))))"*
* **Thử lửa trên 100% dữ liệu thật sống hôm nay (`training/test_real_world_data.py`):**
  1. **Module URL Phishing trên tấn công sống hôm nay:**
     * **OpenPhish LIVE hôm nay (300 link mới):** Bắt **`91.33%`** (`274/300`).
     * **Phishing.Database ACTIVE hôm nay (10.000 link mới):** Bắt **`92.90%`** (`9.290/10.000`).
     * **20 website thật Việt Nam:** An toàn **`100.00%`** (`20/20`).
     * **Tổng hợp LIVE Internet Benchmark (16.749 URL):** **`ROC-AUC = 0.9623` | `F1 = 0.9286`**.
  2. **Module Sparkov (Synthetic) trên 590.540 giao dịch thật IEEE-CIS Vesta:**
     * **Sụp đổ hoàn toàn (`ROC-AUC = 0.4828`, `Recall = 1.31%`)** — chứng minh mô hình học từ dữ liệu giả lập không thể dùng cho giao dịch người thật!
* **Chốt nâng cấp (`training/train_vesta_real.py`):** Huấn luyện lõi thanh toán mới trực tiếp trên **590.540 giao dịch thật của Vesta Corporation (IEEE-CIS, 0% Synthetic)** với 5-Fold Out-Of-Fold Target Encoding + Rolling Velocity 1h/24h + Chronological Future Test (`88.581` GD tương lai thật):
  * **`ROC-AUC = 0.89207` | `PR-AUC = 0.47980` (Lift 13.8x) | `Recall @ 1% FPR = 42.17%` | `Recall @ 5% FPR = 58.19%`**.

## Phần 4: Stress Test sâu & Thử lửa thêm 4 nguồn LIVE hoàn toàn mới
* **User:** *"Test lại cho t"* → *"Thật sự được kq đó không vậy, kiếm thêm vài bộ nữa để test"*
* **Kết quả `training/verify_real_models.py`:**
  * **Vesta thật cứu được:** **\$150,075** (`1% FPR`), **\$185,491** (`2.02% FPR`), **\$240,659** (`5% FPR`) trong tổng \$469,608 thiệt hại gian lận thật; **Cold-Start Unseen `card1`** đạt **`ROC-AUC = 0.9454`**, **Unseen UID** đạt **`0.8678`**; **Mobile `0.9104` / Desktop `0.8947`**; độ trễ Live + SHAP `49 ms`.
  * **URL thật:** **20/20 (`100%`)** kịch bản thực tế Việt Nam thuần ML (`0.55 ms/URL`); bắt **`70.59%`** nhóm siêu tinh vi `HTTPS + .com/.org/.net` trong OpenPhish LIVE hôm nay.
* **Kết quả `training/test_extra_real_datasets.py` (4 nguồn LIVE mới + 3 bài ép lỗi):**
  * **Báo chí VN LIVE RSS hôm nay (`VNExpress/TuoiTre/DanTri`, 208 bài sâu):** An toàn **`100.00%`** (`208/208`, 0 báo nhầm).
  * **HackerNews LIVE hôm nay (87 URL sâu):** An toàn **`95.40%`** (`83/87`; 4 ca báo nhầm là `github.io` và `@` của Mastodon).
  * **CERT.pl LIVE (10.000 tên miền trần không có path):** Bắt **`67.35%`**.
  * **URLhaus LIVE (10.000 host phát tán mã độc):** Bắt **`50.97%`**.
  * **Hard Live Test tổng hợp (`20.295` URL):** **`ROC-AUC = 0.9477`**.
  * **Ép lỗi Vesta thật:** 4 tuần tương lai liên tiếp ổn định `AUC 0.880 – 0.905`; xóa trắng 100% 10 cột V-block vẫn giữ `AUC 0.8844`; xóa trắng 100% lịch sử OOF vẫn giữ `AUC 0.8513`.

## Phần 5: Cập nhật mã nguồn, model Vesta thật & toàn bộ kết quả kiểm định lên GitHub
* **User:** *"push mấy cái mới lên"*
* **Assistant:** Cập nhật `README.md`, `MODEL_EVALUATION_REPORT.md`, `docs/CONVERSATION_LOG.md`, bổ sung `models/vesta_real_bundle.joblib`, `data/vesta_real_ops_samples.parquet` và 4 script kiểm định thực tế (`train_vesta_real.py`, `test_real_world_data.py`, `verify_real_models.py`, `test_extra_real_datasets.py`) vào commit mới và đẩy lên `https://github.com/dothanhtuan1704-droid/MOMO.git`.
