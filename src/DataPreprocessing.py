'''
CHỨC NĂNG: Làm sạch dữ liệu, tính RFM và lưu file data.csv
'''
import pandas as pd

# --- Đọc dữ liệu gốc ---
df = pd.read_csv("data/online_retail.csv")

print("=== THÔNG TIN BAN ĐẦU ===")
print(df.info())
print("Số dòng ban đầu:", df.shape[0])

# --- Làm sạch dữ liệu ---
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# --- Tính RFM ---
latest_date = df["InvoiceDate"].max()
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (latest_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()

rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "TotalPrice": "Monetary"
}, inplace=True)

print("\n=== MẪU DỮ LIỆU RFM ===")
print(rfm.head())

# --- Lưu ra file ---
rfm.to_csv("data/data.csv", index=False)
print("\n✅ Đã lưu dữ liệu RFM vào file: data/data.csv")
