# 🛍️ HỆ THỐNG PHÂN CỤM KHÁCH HÀNG DỰA TRÊN ĐẶC ĐIỂM GIAO DỊCH
### ỨNG DỤNG TRONG TỐI ƯU HÓA CHIẾN LƯỢC MARKETING

Dự án xây dựng một hệ thống Web **Flask** cho phép phân cụm khách hàng dựa trên bộ đặc trưng **RFM mở rộng** và thuật toán **K-Means**, đồng thời hỗ trợ **dự đoán cụm mới** thông qua giao diện trực quan.

Hệ thống giúp doanh nghiệp:
- Hiểu rõ phân khúc khách hàng.
- Tối ưu hóa chiến lược marketing cho từng nhóm.
- Trực quan hóa dữ liệu và cụm bằng biểu đồ.

---

🚀 1. Tính năng chính

### ✔ Upload file CSV để phân cụm
Người dùng tải lên dữ liệu giao dịch đã được xử lý RFM (Recency, Frequency, Monetary, TotalQuantity, AvgUnitPrice).
Hệ thống tự động chuẩn hóa → phân cụm → hiển thị bảng kết quả và biểu đồ.

### ✔ Trực quan hóa cụm
- Biểu đồ **đặc trưng trung bình từng cụm**
- Biểu đồ **Scatter Frequency – Monetary**
(được vẽ trong `/results` bằng matplotlib và seaborn)

### ✔ Dự đoán cụm khách hàng mới
Nhập 5 giá trị RFM → hệ thống trả về cụm tương ứng + mô tả chi tiết cụm.

### ✔ Mô tả chuyên sâu từng cụm
Theo cấu trúc đã khai báo trong `CLUSTER_DESCRIPTION` của app.py.


📂 2. Cấu trúc dự án

    project/
    │
    ├── src/
    │ ├── app.py       # Flask web app
    │ ├── uploads/     # File CSV người dùng upload
    │ ├── views/       # Templates (HTML)
    │ │ ├── base.html
    │ │ ├── index.html
    │ │ ├── results.html
    │ │ └── predict.html
    │ ├── ClusteringModel.py    # Huấn luyện & lưu mô hình KMeans
    │ ├── DataPreprocessing.py  # Tiền xử lý + tính RFM
    │ ├── ElbowPlot.py          # Elbow Method để chọn k
    │ ├── Evaluation.py         # Đánh giá mô hình
    │ └── Chart.py              # Các hàm vẽ biểu đồ
    │
    ├── models/
    │ ├── scaler.pkl            # StandardScaler đã huấn luyện
    │ └── kmeans_model.pkl      # Mô hình KMeans (k=4)
    │
    ├── images/
    │ ├── bg/                   # Background
    │ ├── cluster_summary.png
    │ └── scatter.png
    │
    ├── data/
    │ ├── online_retail.csv
    │ ├── datafinal.csv              # Dữ liệu đã xử lý & loại outlier
    │ └── rfm_clustered_final.csv    # Dữ liệu đã gán nhãn cụm
    │
    └── README.md


🔧 3. Công nghệ sử dụng

- Flask – xây dựng giao diện web
- Pandas / NumPy – xử lý dữ liệu
- Scikit-learn – StandardScaler + KMeans
- **Matplotlib / Seaborn** – trực quan hóa
- Bootstrap 5 – giao diện hiện đại
- Joblib – lưu mô hình


🔍 4. Quy trình hoạt động của hệ thống

## Bước 1 — Tiền xử lý dữ liệu
Theo file DataPreprocessing.py :contentReference[oaicite:3]{index=3}
- Làm sạch dữ liệu Online Retail
- Tính RFM mở rộng
- Loại outlier theo percentile
- Lưu file `datafinal.csv`

## Bước 2 — Huấn luyện mô hình
Theo file ClusteringModel.py :contentReference[oaicite:4]{index=4}
- Chuẩn hóa dữ liệu bằng StandardScaler
- Huấn luyện KMeans (k=4)
- Lưu `scaler.pkl` và `kmeans_model.pkl`

## Bước 3 — Chạy hệ thống Flask
Theo file app.py :contentReference[oaicite:5]{index=5}
Các chức năng gồm:

### `/` – Trang chủ
Giới thiệu hệ thống + upload file CSV.

### `/upload`
Nhận file CSV → lưu vào thư mục uploaded → chuyển đến `/results`.

### `/results`
- Đọc file CSV mới
- Chuẩn hóa theo scaler
- Dự đoán cụm bằng KMeans
- Tính bảng thống kê
- Vẽ 2 biểu đồ:
    - cluster_summary.png
    - scatter.png

### `/predict`
Nhập RFM → trả kết quả cụm + mô tả cụm.


🧠 5. Bộ mô tả cụm chuẩn hóa (theo app.py)

CLUSTER_DESCRIPTION = {
    0: "Cụm 0 – Khách hàng giá trị nhưng đang rủi ro: từng chi tiêu cao nhưng lâu không quay lại, tần suất thấp.",
    1: "Cụm 1 – Khách hàng hoạt động đều đặn: tần suất và giá trị mua trung bình, chiếm tỷ lệ lớn nhất.",
    2: "Cụm 2 – Khách hàng không hoạt động: thời gian quay lại rất lâu, tần suất thấp, chi tiêu thấp.",
    3: "Cụm 3 – Khách hàng giá trị cao: mua thường xuyên, số lượng lớn, chi tiêu mạnh và quay lại nhanh."
}


▶️ 6. Hướng dẫn chạy hệ thống

1️⃣ Cài môi trường

    pip install -r requirements.txt

Hoặc tối thiểu:
    pip install flask pandas numpy scikit-learn matplotlib seaborn joblib

2️⃣ Chạy Flask

Từ thư mục /src:
    python app.py

Hệ thống chạy mặc định tại:
    http://127.0.0.1:5000/