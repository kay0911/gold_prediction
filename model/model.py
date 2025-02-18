import tensorflow as tf
from model.layers.lstm_cell import LSTM

@tf.keras.utils.register_keras_serializable(package="Custom") 
class LSTMModel(tf.keras.Model):
    def __init__(self, units: int, input_length: int, prediction_length: int, **kwargs):
        """
        Khởi tạo mô hình LSTM cho dự đoán giá vàng.
        
        Tham số:
        - units (int): Số lượng đơn vị ẩn trong tầng LSTM.
        - input_length (int): Độ dài chuỗi đầu vào (N ngày).
        - prediction_length (int): Số ngày dự đoán (M ngày).
        """
        super(LSTMModel, self).__init__(**kwargs)
        self.units = units
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.lstm = LSTM(units, inp_shape=1)  # LSTM cho dự đoán chuỗi
        self.lstm.build(None)  # Khởi tạo weights cho LSTM
        
        # Thêm 2 lớp Dense như yêu cầu
        self.dense = tf.keras.layers.Dense(64, activation='relu')  # Lớp Dense đầu tiên
        
        # Lớp cuối cùng để dự đoán giá vàng cho M ngày
        self.prediction_layer = tf.keras.layers.Dense(prediction_length, activation='linear')

    def call(self, prices: tf.Tensor) -> tf.Tensor:
        """
        Truyền dữ liệu qua mô hình để dự đoán giá vàng.

        Tham số:
        - prices (tf.Tensor): Tensor chứa giá vàng lịch sử, kích thước (batch_size, input_length).
        
        Trả về:
        - Tensor dự đoán giá vàng, kích thước (batch_size, prediction_length).
        """
        batch_size = tf.shape(prices)[0]
        
        # Khởi tạo (hidden_state và context_state)
        pre_layer = tf.stack([
            tf.zeros([batch_size, self.lstm.units]),
            tf.zeros([batch_size, self.lstm.units])
        ])

        # Đưa từng giá vàng qua LSTM để học chuỗi
        for i in range(self.input_length):
            price = prices[:, i:i+1]  # Lấy giá vàng tại thời điểm i
            pre_layer = self.lstm(pre_layer, price)

        h, _ = tf.unstack(pre_layer)

        # Truyền qua các lớp Dense
        x = self.dense(h)

        # Dự đoán giá vàng trong M ngày
        return self.prediction_layer(x)  # Mỗi ngày dự đoán là 1 giá trị liên tiếp

    def get_config(self) -> dict:
        """
        Lưu cấu hình của mô hình để tái sử dụng sau này.
        
        Trả về:
        - dict: Cấu hình của mô hình bao gồm số units, độ dài chuỗi đầu vào và độ dài dự đoán.
        """
        config = super(LSTMModel, self).get_config()
        config.update({
            "units": self.units,
            "input_length": self.input_length,
            "prediction_length": self.prediction_length,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        """
        Tải lại mô hình từ file cấu hình.
        
        Tham số:
        - config (dict): Cấu hình đã lưu của mô hình.
        
        Trả về:
        - LSTMModel: Mô hình đã được khôi phục.
        """
        return cls(**config)
