# PIPELINE: FreqMedCLIP (Smart Single-Stream)

## 1. Input (Đầu vào)
* **Hình ảnh ($I$):** Ảnh y tế (X-ray, CT, Ultrasound...).
* **Văn bản ($T$):** Câu lệnh nhắc (Prompt), ví dụ: "A benign breast tumor" (Một khối u vú lành tính).

## 2. Feature Extraction (Trích xuất đặc trưng)
Thay vì chạy 2 mô hình nặng nề song song, ta chỉ dùng 1 mô hình BiomedCLIP nhưng lấy dữ liệu ở 2 trạm dừng khác nhau:

* **Nhánh ngữ nghĩa (LF - Low Frequency):**
    * Dữ liệu đi hết qua BiomedCLIP Image Encoder.
    * Lấy output ở lớp cuối cùng.
    * **Ý nghĩa:** Chứa thông tin "Đây là cái gì?" (Gan, Phổi, U...). Đây là thế mạnh của MedCLIP-SAMv2 gốc.
* **Nhánh chi tiết (HF - High Frequency):**
    * Lấy output ở các lớp nông (ví dụ: Layer 3 hoặc 4 của Encoder). Tại đây ảnh vẫn còn giữ được các đường nét, chưa bị mờ đi.
    * **Injection (Tiêm tần số):** Dùng thuật toán Wavelet (như trong bài FMISeg 2022) tách biên của ảnh gốc, nén nhỏ lại và cộng vào feature này.
    * **Ý nghĩa:** Chứa thông tin "Biên của vật thể nằm chính xác ở pixel nào?".

## 3. The "Smart" Fusion (Pha trộn thông minh)
Đây là phần thay đổi lớn nhất so với đề xuất cũ. Ta xử lý theo quy trình **Coarse-to-Fine** (Từ thô đến tinh):

* **Bước 3.1: Định vị thô (Coarse Localization)**
    * Đưa Văn bản ($T$) và Feature Ngữ nghĩa (LF) vào module M2IB (như MedCLIP-SAMv2 cũ).
    * **Kết quả:** Ra một bản đồ nhiệt (Saliency Map) mờ mờ, khoanh vùng được vị trí khối u nhưng biên chưa chuẩn.
* **Bước 3.2: Tinh chỉnh biên (Frequency Refinement)**
    * Lấy bản đồ thô ở trên làm "đèn pin" soi vào Feature Chi tiết (HF).
    * Ta dùng cơ chế Attention hoặc FFBI đơn giản hóa: *"Tại vùng mà văn bản đã chỉ ra (LF), hãy tìm các cạnh sắc nét nhất trong HF và bám theo đó"*.
    * **Kết quả:** Bản đồ nhiệt mới ($S_{fused}$) vừa đúng vị trí (nhờ Text/LF), vừa sắc nét (nhờ HF).

## 4. SAM Refinement (Bước cuối)
Bước này giữ nguyên như MedCLIP-SAMv2:
* Từ bản đồ nhiệt $S_{fused}$ sắc nét kia, ta tạo ra một Bounding Box (hộp bao) hoặc các điểm (Points) chuẩn xác hơn nhiều so với bản gốc.
* Đưa Box/Point này vào SAM để mô hình này cắt lớp (segment) cuối cùng.

### Tại sao Pipeline này tốt hơn?
1.  **Giải quyết vấn đề "Lệch pha":**
    * Văn bản ("khối u") chỉ nói chuyện với Feature Ngữ nghĩa (LF) ở Bước 3.1. Nó không bị ép phải hiểu Feature Tần số (HF) vốn dĩ rất khó hiểu. Điều này tránh được rủi ro sai lệch.
2.  **Nhẹ hơn (Faster):**
    * Bạn chỉ chạy BiomedCLIP Encoder 1 lần duy nhất. Phần tính toán Wavelet và phép cộng HF rất nhẹ, không tốn đáng kể tài nguyên tính toán.
3.  **Chính xác hơn (Better Boundaries):**
    * MedCLIP-SAMv2 thường bị chê là tạo ra mask bị "tròn quá" hoặc không bám sát các cạnh gồ ghề của khối u. Việc thêm HF (tần số cao) giúp ép mô hình phải chú ý đến các chi tiết gồ ghề đó.