import matplotlib.pyplot as plt
from data import Dataset
from model import LSTMModel
import numpy as np
print("Step 1: Loading data...")

data = Dataset()
ticker = 'GC=F'
years = 20
gold_data = data.load_data(ticker, years)

print("Data downloaded successfully")

print("Step 2: Training...")

gold_data = gold_data[['Close']].values
normalized_data = data.build_normalized(gold_data)

input_length = 14  # Dùng 14 ngày trước đó
prediction_length = 7  # Dự đoán 7 ngày tiếp theo
units = 128  # Số đơn vị trong LSTM

X, y = data.create_dataset(normalized_data, input_length, prediction_length)
(X_train, y_train), (X_test, y_test) = data.split_data(X, y)
# Tạo mô hình
model = LSTMModel(units=units, input_length=input_length, prediction_length=prediction_length)

# Tiền xử lý dữ liệu và huấn luyện mô hình
# (Giả sử `X_train` và `y_train` là dữ liệu huấn luyện)
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=10, batch_size=32)

print("Step 3: Model results...")

# Vẽ đồ thị kết quả huấn luyện
plt.figure(figsize=(10, 5))

# Vẽ loss của dữ liệu huấn luyện và validation
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Thêm nhãn và tiêu đề cho đồ thị
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Lưu đồ thị vào file ảnh
plot_path = "result/training_plot.png"
plt.savefig(plot_path)
print(f"Training plot saved to {plot_path}")

# Hiển thị đồ thị
plt.show()

print("Step 4: Prediction...")

# Lấy 14 ngày gần nhất từ dữ liệu đầu vào
data_predict = X_test[-1].reshape(1, input_length)
last_data = data.inverse_normalized(data_predict)

# Dự đoán giá vàng trong 7 ngày tiếp theo
prediction = model(data_predict)
prediction_data = data.inverse_normalized(np.array(prediction))

# Tạo dữ liệu cho biểu đồ (14 ngày gần nhất và 7 ngày dự đoán)
days = np.arange(1, input_length + prediction_length + 1)  # Số ngày từ 1 đến N + M
actual_data = np.concatenate((last_data.flatten(), prediction_data.flatten()))  # Kết hợp last_data và prediction_data

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.plot(days[:input_length], last_data.flatten(), label='Last 14 Days', color='blue')
plt.plot(days[input_length:], prediction_data.flatten(), label='Predicted Next 7 Days', color='orange')
# Vẽ đường nối giữa điểm cuối của last_data và đầu của prediction_data bằng nét đứt
plt.plot([days[input_length-1], days[input_length]], [last_data.flatten()[-1], prediction_data.flatten()[0]], 
         linestyle='--', color='gray', label='Prediction Continuity')
# Thêm nhãn và tiêu đề cho đồ thị
plt.xlabel('Days')
plt.ylabel('Gold Price')
plt.title('Gold Price Prediction: Last 14 Days and Next 7 Days')
plt.legend()

# Lưu đồ thị vào file ảnh
plot_path = "result/gold_price_prediction_plot.png"
plt.savefig(plot_path)
print(f"Prediction plot saved to {plot_path}")

# Hiển thị đồ thị
plt.show()

