import pandas as pd
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ==============================
# 1. Load dữ liệu và mô hình đã lưu
# ==============================
df = pd.read_csv("../data/datafinal.csv")
scaler = joblib.load("../models/scaler.pkl")
kmeans = joblib.load("../models/kmeans_model.pkl")  # chắc chắn là k=4

features = ["Recency", "Frequency", "Monetary", "TotalQuantity", "AvgUnitPrice"]
X_scaled = scaler.transform(df[features])

# Gán nhãn cụm vào dataframe
df["Cluster"] = kmeans.labels_

# ==============================
# 2. Tính 3 chỉ số đánh giá
# ==============================
inertia = kmeans.inertia_
silhouette = silhouette_score(X_scaled, kmeans.labels_)
db_index = davies_bouldin_score(X_scaled, kmeans.labels_)

# ==============================
# 3. Kích thước và tỷ lệ các cụm
# ==============================
cluster_sizes = df["Cluster"].value_counts().sort_index()
cluster_percent = (cluster_sizes / len(df) * 100).round(1)

# ==============================
# 4. Bảng đặc trưng trung bình từng cụm
# ==============================
summary = df.groupby("Cluster")[features].mean().round(2)

# ==============================
# 5. IN KẾT QUẢ ĐẸP ĐỂ COPY VÀO BÁO CÁO
# ==============================
print("="*70)
print("                KẾT QUẢ ĐÁNH GIÁ PHÂN CỤM (k=4)")
print("="*70)
print(f"Số khách hàng tổng cộng      : {len(df):,}")
print(f"Inertia (WCSS)               : {inertia:,.2f}")
print(f"Silhouette Score             : {silhouette:.3f} ")
print(f"Davies-Bouldin Index         : {db_index:.3f}")
print("-"*70)

print("\nKÍCH THƯỚC VÀ TỶ LỆ CÁC CỤM")
print("-"*50)
for cluster_id in cluster_sizes.index:
    print(f"Cụm {cluster_id} : {cluster_sizes[cluster_id]:,} khách hàng → {cluster_percent[cluster_id]}%")

print("\nBẢNG ĐẶC TRƯNG TRUNG BÌNH TỪNG CỤM")
print("-"*80)
print(summary.round(2))
print("-"*80)

# ==============================
# 6. (Tùy chọn) Lưu lại file CSV đã có Cluster để vẽ biểu đồ sau này
# ==============================
df.to_csv("../data/rfm_clustered_final.csv", index=False)
print("\nĐã lưu file có cột Cluster → ../data/rfm_clustered_final.csv")