import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_summary(df: pd.DataFrame):
    """Vẽ biểu đồ trung bình đặc trưng RFM của từng cụm"""
    summary = df.groupby("Cluster")[["Recency", "Frequency", "Monetary", "TotalQuantity", "AvgUnitPrice"]].mean()
    summary.plot(kind="bar", figsize=(10, 6), colormap="tab10")
    plt.title("Đặc trưng trung bình của từng cụm khách hàng")
    plt.ylabel("Giá trị trung bình (chuẩn hoá)")
    plt.tight_layout()
    plt.show()


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    """Vẽ scatter plot thể hiện sự phân bố khách hàng"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue="Cluster", palette="Set2", s=50)
    plt.title(f"Phân bố khách hàng theo {x} và {y}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../data/rfm_clustered.csv")
    plot_cluster_summary(df)
    plot_scatter(df, "Frequency", "Monetary")
