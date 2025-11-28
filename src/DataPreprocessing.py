import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Đọc và làm sạch dữ liệu gốc"""
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"[INFO] Dữ liệu gốc: {len(df):,} dòng")

    # Loại bỏ lỗi / dữ liệu không hợp lệ
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print(f"[INFO] Sau làm sạch: {len(df):,} dòng | Khách hàng: {df['CustomerID'].nunique():,}")

    return df


def compute_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính các chỉ số RFM + đặc trưng mở rộng"""
    latest_date = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,   # Recency
        'InvoiceNo': 'nunique',                                  # Frequency
        'TotalPrice': 'sum',                                     # Monetary
        'Quantity': 'sum',                                       # Tổng SL mua
        'UnitPrice': 'mean',                                     # Giá TB
        'Country': lambda x: x.mode()[0]                         # Quốc gia chính
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary',
        'Quantity': 'TotalQuantity',
        'UnitPrice': 'AvgUnitPrice'
    }).reset_index()

    return rfm


def save_data(df: pd.DataFrame, output_path: str):
    """Lưu dữ liệu ra file CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✅] Đã lưu dữ liệu tiền xử lý tại: {output_path}")


if __name__ == "__main__":
    input_path = "../data/online_retail.csv"
    output_path = "../data/datafinal.csv"

    df = load_and_clean_data(input_path)
    rfm = compute_rfm_features(df)
    save_data(rfm, output_path)
