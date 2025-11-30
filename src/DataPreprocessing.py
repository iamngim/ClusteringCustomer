import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print("[1/4] Đang đọc dữ liệu gốc...")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"    → Dữ liệu gốc: {len(df):,} giao dịch")

    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    print(f"    → Sau làm sạch: {len(df):,} giao dịch | {df['CustomerID'].nunique():,} khách hàng")
    return df


def compute_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/4] Đang tính RFM...")
    latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary',
        'Quantity': 'TotalQuantity',
        'UnitPrice': 'AvgUnitPrice'
    }).reset_index()

    print(f"    → Tạo RFM cho {len(rfm):,} khách hàng")
    return rfm


def remove_outliers_percentile(rfm: pd.DataFrame) -> pd.DataFrame:
    print("Loại bỏ outlier")
    before = len(rfm)

    # Cách chuẩn nhất cho Online Retail: loại top 2-3% ở Monetary và Frequency
    rfm = rfm[rfm['Monetary'] <= rfm['Monetary'].quantile(0.975)]  # loại top 2.5%
    rfm = rfm[rfm['Frequency'] <= rfm['Frequency'].quantile(0.975)]  # loại top 2.5%
    rfm = rfm[rfm['TotalQuantity'] <= rfm['TotalQuantity'].quantile(0.975)]

    # Loại khách hàng có giá trung bình quá cao (loại bỏ hoàn toàn trường hợp 2033.10)
    rfm = rfm[rfm['AvgUnitPrice'] <= 20]  # 99.99% khách hàng có giá TB < 20

    after = len(rfm)
    print(f"    → Loại bỏ {before - after} khách hàng bất thường → còn {after:,} khách hàng")
    return rfm


def save_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[4/4] ĐÃ LƯU datafinal.csv → {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("BẮT ĐẦU TIỀN XỬ LÝ")
    print("=" * 80)

    raw = load_and_clean_data("../data/online_retail.csv")
    rfm = compute_rfm_features(raw)
    rfm_clean = remove_outliers_percentile(rfm)
    save_data(rfm_clean, "../data/datafinal.csv")

    print("=" * 80)