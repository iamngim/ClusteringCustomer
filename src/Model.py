import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Đọc dữ liệu RFM ---
rfm = pd.read_csv("rfm_data.csv")

print("=== DỮ LIỆU RFM SAU TIỀN XỬ LÝ ===")
print(rfm.head())

# --- Chuẩn hóa dữ liệu ---
features = ["Recency", "Frequency", "Monetary"]
X = rfm[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Chọn số cụm bằng Elbow Method ---
inertias = []
K = range(2, 8)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker="o")
plt.title("Elbow Method - Chọn số cụm k")
plt.xlabel("Số cụm (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# --- Áp dụng KMeans (k=4) ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n=== SỐ LƯỢNG KHÁCH HÀNG TRONG TỪNG CỤM ===")
print(rfm["Cluster"].value_counts())

cluster_stats = rfm.groupby("Cluster")[features].mean().round(2)
print("\n=== TRUNG BÌNH RFM THEO CỤM ===")
print(cluster_stats)

# --- Trực quan hóa cụm ---
plt.figure(figsize=(8,6))
sns.scatterplot(x="Recency", y="Monetary", hue="Cluster", data=rfm, palette="Set2")
plt.title("Phân cụm khách hàng theo Recency & Monetary")
plt.xlabel("Recency (ngày kể từ lần mua gần nhất)")
plt.ylabel("Monetary (tổng chi tiêu)")
plt.show()

# --- Gợi ý chiến lược marketing ---
summary = cluster_stats.copy()
summary["Chiến lược Marketing"] = [
    "Khách hàng trung thành – duy trì bằng ưu đãi định kỳ",
    "Khách hàng mới – khuyến mãi chào mừng, remarketing",
    "Khách hàng tiềm năng – cá nhân hóa quảng cáo, upsell",
    "Khách hàng sắp rời bỏ – tái kích hoạt bằng giảm giá mạnh"
]

print("\n=== GỢI Ý CHIẾN LƯỢC MARKETING ===")
print(summary)

# --- Lưu kết quả ---
rfm.to_csv("customer_segmentation_result.csv", index=False)
print("\n✅ Đã lưu kết quả phân cụm vào: customer_segmentation_result.csv")
