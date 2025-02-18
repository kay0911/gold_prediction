import tensorflow as tf

class LSTM(tf.keras.layers.Layer):
    def __init__(self, units: int, inp_shape: int):
        """
        Khởi tạo lớp LSTM tùy chỉnh.
        
        Tham số:
        - units (int): Số lượng đơn vị ẩn trong tầng LSTM.
        - inp_shape (int): Kích thước đầu vào của dữ liệu.
        """
        super().__init__()  # ✅ Gọi super()
        self.units = units
        self.inp_shape = inp_shape

    def build(self, input_shape: tuple):
        """
        Khởi tạo trọng số cho mạng LSTM.
        
        Tham số:
        - input_shape (tuple): Kích thước của đầu vào.
        """
        # ✅ Khởi tạo trọng số đúng cách
        self.W_i = self.add_weight(name="W_i", shape=(self.inp_shape, self.units))
        self.W_f = self.add_weight(name="W_f", shape=(self.inp_shape, self.units))
        self.W_o = self.add_weight(name="W_o", shape=(self.inp_shape, self.units))
        self.W_c = self.add_weight(name="W_c", shape=(self.inp_shape, self.units))

        self.U_i = self.add_weight(name="U_i", shape=(self.units, self.units))
        self.U_f = self.add_weight(name="U_f", shape=(self.units, self.units))
        self.U_o = self.add_weight(name="U_o", shape=(self.units, self.units))
        self.U_c = self.add_weight(name="U_c", shape=(self.units, self.units))

    def call(self, pre_layer: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """
        Xử lý dữ liệu qua tầng LSTM.
        
        Tham số:
        - pre_layer (tf.Tensor): Tensor chứa trạng thái ẩn (h) và bộ nhớ (c) từ bước thời gian trước đó.
        - x (tf.Tensor): Đầu vào hiện tại của tầng LSTM tại bước thời gian đang xét.
        
        Trả về:
        - Tensor chứa trạng thái mới của LSTM sau khi xử lý.
        """
        pre_h, pre_c = tf.unstack(pre_layer)

        # ✅ Sử dụng từng trọng số riêng biệt
        i_t = tf.nn.sigmoid(tf.matmul(x, self.W_i) + tf.matmul(pre_h, self.U_i))
        f_t = tf.nn.sigmoid(tf.matmul(x, self.W_f) + tf.matmul(pre_h, self.U_f))
        o_t = tf.nn.sigmoid(tf.matmul(x, self.W_o) + tf.matmul(pre_h, self.U_o))
        n_c_t = tf.nn.tanh(tf.matmul(x, self.W_c) + tf.matmul(pre_h, self.U_c))

        c = f_t * pre_c + i_t * n_c_t  # ✅ Sửa tf.multiply() thành toán tử *
        h = o_t * tf.nn.tanh(c)

        return tf.stack([h, c])