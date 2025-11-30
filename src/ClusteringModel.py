# ================================================================
# File: ClusteringModel.py (ĐÃ ĐƯỢC NÂNG CẤP - CÓ LƯU MÔ HÌNH)
# Mục tiêu: Phân cụm khách hàng + LƯU mô hình để tái sử dụng
# ================================================================

import os
import warnings
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# ---------------------------------------------------------------
# 1. HÀM XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu đã tiền xử lý"""
    df = pd.read_csv(file_path)
    print(f"[INFO] Đã load {len(df):,} khách hàng.")
    return df


# ---------------------------------------------------------------
# 2. HÀM XÁC ĐỊNH SỐ CỤM TỐI ƯU (Elbow Method)
# ---------------------------------------------------------------
def find_optimal_clusters(X, max_k=10):
    """Hiển thị biểu đồ Elbow để chọn số cụm K tối ưu"""
    inertia = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.xlabel("Số cụm (k)")
    plt.ylabel("Inertia")
    plt.title("Biểu đồ Elbow xác định số cụm tối ưu")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 3. HÀM PHÂN CỤM BẰNG KMEANS (TRẢ VỀ CẢ MODEL + SCALER)
# ---------------------------------------------------------------
def run_kmeans(df: pd.DataFrame, features: list, k: int = 4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled)

    print(f"[ĐÃ PHÂN CỤM] Thành {k} nhóm khách hàng.")
    return df, model, scaler


# ---------------------------------------------------------------
# 4. HÀM PHÂN TÍCH CHI TIẾT CÁC CỤM
# ---------------------------------------------------------------
def analyze_clusters(df: pd.DataFrame, features: list):
    # (giữ nguyên như cũ)
    summary = df.groupby("Cluster")[features].mean().round(2)
    counts = df["Cluster"].value_counts()

    print("\nTỔNG QUAN PHÂN CỤM")
    print(summary)
    print("\nSố lượng khách hàng trong từng cụm:")
    print(counts.sort_index())

    print("\nPHÂN TÍCH CHI TIẾT")
    for cluster_id in summary.index:
        data = summary.loc[cluster_id]
        size = counts[cluster_id]
        print(f"\n--- Cụm {cluster_id} ({size} khách hàng) ---")
        print(f"Recency TB: {data['Recency']}")
        print(f"Frequency TB: {data['Frequency']}")
        print(f"Monetary TB: {data['Monetary']:.0f}")
        print(f"TotalQuantity TB: {data['TotalQuantity']:.0f}")
        print(f"AvgUnitPrice TB: {data['AvgUnitPrice']:.2f}")

        if data['Monetary'] > summary['Monetary'].mean() and data['Frequency'] > summary['Frequency'].mean():
            desc = "Khách hàng VIP / Trung thành - chi tiêu và tần suất mua cao."
        elif data['Recency'] < summary['Recency'].mean() and data['Frequency'] > summary['Frequency'].mean():
            desc = "Khách hàng tiềm năng - mua gần đây và khá thường xuyên."
        elif data['Recency'] > summary['Recency'].mean() and data['Frequency'] < summary['Frequency'].mean():
            desc = "Khách hàng không hoạt động - lâu chưa quay lại, tần suất thấp."
        else:
            desc = "Nhóm khách hàng trung bình hoặc đặc biệt."

        print("→ Nhận xét:", desc)

    return summary


# ---------------------------------------------------------------
# 5. HÀM LƯU DỮ LIỆU ĐÃ PHÂN CỤM
# ---------------------------------------------------------------
def save_clustered_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[ĐÃ LƯU] Dữ liệu phân cụm → {output_path}")


def save_model_artifacts(scaler, model, model_dir="../models"):
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(model, os.path.join(model_dir, "kmeans_model.pkl"))

    print(f"[ĐÃ LƯU] Scaler → {model_dir}/scaler.pkl")
    print(f"[ĐÃ LƯU] Mô hình KMeans → {model_dir}/kmeans_model.pkl")


def main():
    input_path = "../data/datafinal.csv"
    output_path = "../data/rfm_clustered.csv"
    features = ["Recency", "Frequency", "Monetary", "TotalQuantity", "AvgUnitPrice"]

    df = load_data(input_path)

    # Phân cụm và lấy cả model + scaler
    df_clustered, kmeans_model, scaler = run_kmeans(df, features, k=4)

    analyze_clusters(df_clustered, features)
    save_clustered_data(df_clustered, output_path)

    # LƯU MÔ HÌNH ĐỂ DÙNG LẠI CHO DATA MỚI
    save_model_artifacts(scaler, kmeans_model)

    print("\nHOÀN TẤT! Mô hình đã được lưu.")


if __name__ == "__main__":
    main()