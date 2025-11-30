# ================================================================
# File: ElbowPlot.py
# Mục đích: Vẽ biểu đồ Elbow để chọn k tối ưu và lưu hình
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 1. Load dữ liệu đã tiền xử lý
df = pd.read_csv("../data/datafinal.csv")
features = ["Recency", "Frequency", "Monetary", "TotalQuantity", "AvgUnitPrice"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

inertia = []
k_range = range(2, 13)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"k = {k} → Inertia = {kmeans.inertia_:,.2f}")

# 5. Vẽ biểu đồ Elbow (đẹp như báo cáo)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linewidth=2.5, markersize=8, color='#2c7bb6')
plt.title('Biểu đồ Elbow - Xác định số cụm tối ưu', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Số lượng cụm (k)', fontsize=14)
plt.ylabel('Tổng bình phương khoảng cách trong cụm (Inertia)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_range)

# Đánh dấu điểm k=4 (điểm khuỷu tay rõ nhất trên tập dữ liệu Online Retail)
plt.axvline(x=4, color='red', linestyle='--', linewidth=2, label='k = 4 (điểm khuỷu tay)')
plt.legend(fontsize=12)

# Lưu hình chất lượng cao để chèn vào báo cáo
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/elbow_plot.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nBiểu đồ Elbow đã được lưu tại: ../images/elbow_plot.png")
print("Bạn có thể chèn hình này vào báo cáo với chú thích: Hình 3.5 - Biểu đồ Elbow xác định số cụm tối ưu")