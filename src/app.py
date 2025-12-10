from flask import Flask, render_template, request, redirect
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# FIX BASE PATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)   # cha của /src

MODEL_FOLDER = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, "uploads")   # uploads nằm trong src/
STATIC_FOLDER = os.path.join(BASE_DIR, "images")       # images/ ở ngoài src/
VIEWS_FOLDER = os.path.join(CURRENT_DIR, "views")

CLUSTER_DESCRIPTION = {
    0: "Cụm 0 – Khách hàng giá trị nhưng đang rủi ro: từng chi tiêu cao nhưng lâu không quay lại, tần suất thấp.",
    1: "Cụm 1 – Khách hàng hoạt động đều đặn: tần suất và giá trị mua ở mức trung bình, chiếm tỷ lệ lớn nhất.",
    2: "Cụm 2 – Khách hàng không hoạt động: thời gian quay lại rất lâu, tần suất thấp, chi tiêu thấp.",
    3: "Cụm 3 – Khách hàng giá trị cao: mua thường xuyên, số lượng lớn, chi tiêu mạnh và quay lại nhanh."
}


app = Flask(
    __name__,
    template_folder=VIEWS_FOLDER,
    static_folder=STATIC_FOLDER
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load mô hình
scaler = joblib.load(os.path.join(MODEL_FOLDER, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODEL_FOLDER, "kmeans_model.pkl"))

FEATURES = ["Recency", "Frequency", "Monetary", "TotalQuantity", "AvgUnitPrice"]


# Trang chủ
@app.route("/")
def home():
    return render_template("index.html")


# Upload file CSV
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file:
        return "❌ Không tìm thấy file!"

    file.save(os.path.join(UPLOAD_FOLDER, "data.csv"))
    return redirect("/results")


# Kết quả phân cụm
@app.route("/results")
def results():
    csv_path = os.path.join(UPLOAD_FOLDER, "data.csv")

    if not os.path.exists(csv_path):
        return "<h2>Bạn chưa upload file!</h2>"

    df = pd.read_csv(csv_path)

    # Chuẩn hoá và phân cụm
    X_scaled = scaler.transform(df[FEATURES])
    df["Cluster"] = kmeans.predict(X_scaled)

    summary = df.groupby("Cluster")[FEATURES].mean().round(2)

    # ============= VẼ BIỂU ĐỒ 1 =============
    plt.figure(figsize=(10, 5))
    summary.plot(kind="bar", figsize=(10, 5))
    plt.title("Đặc trưng trung bình theo từng cụm")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, "cluster_summary.png"))
    plt.close()

    # ============= VẼ BIỂU ĐỒ 2 =============
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Frequency", y="Monetary", hue="Cluster", palette="tab10")
    plt.title("Scatter Frequency - Monetary")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_FOLDER, "scatter.png"))
    plt.close()

    return render_template(
        "results.html",
        summary=summary.to_dict(),
        clusters=sorted(summary.index)
    )


# Dự đoán khách hàng mới
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    cluster_desc = None

    if request.method == "POST":
        data = [
            float(request.form["Recency"]),
            float(request.form["Frequency"]),
            float(request.form["Monetary"]),
            float(request.form["TotalQuantity"]),
            float(request.form["AvgUnitPrice"]),
        ]

        X_scaled = scaler.transform([data])
        result = int(kmeans.predict(X_scaled)[0])
        cluster_desc = CLUSTER_DESCRIPTION.get(result, "Không có mô tả cho cụm này.")

    return render_template("predict.html", result=result, cluster_desc=cluster_desc)



# Run Flask
if __name__ == "__main__":
    print("Base DIR:", BASE_DIR)
    print("Model DIR:", MODEL_FOLDER)
    print("Uploads DIR:", UPLOAD_FOLDER)
    print("Views DIR:", VIEWS_FOLDER)
    print("Static DIR:", STATIC_FOLDER)

    app.run(debug=True)
