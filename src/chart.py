import matplotlib.pyplot as plt

# Dữ liệu tổng quan
labels = ["Tổng số giao dịch", "Số khách hàng có mã định danh", "Số quốc gia khác nhau", "Tỷ lệ giao dịch bị thiếu CustomerID"]
values = [541909, 4372, 38, 24.93]
colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"]

# Vẽ biểu đồ
plt.figure(figsize=(10,6))
bars = plt.bar(labels, values, color=colors)

# Gắn giá trị trên cột
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (bar.get_height()*0.01),
             f"{bar.get_height():,.2f}", ha='center', va='bottom', fontsize=10)

# Tiêu đề ngắn gọn
plt.title("Tổng quan dữ liệu", fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right', fontsize=11)

plt.tight_layout()
plt.show()
