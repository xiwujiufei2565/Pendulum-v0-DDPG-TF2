# 导入相关依赖包
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
import numpy

# 常量定义
HIDDEN1_UNITS = 50 # 第一层神经元
HIDDEN2_UNITS = 50 # 第二层神经元
HIDDEN3_UNITS = 16 # 第三层神经元

class Critic:
    def __init__(self, state_dim, action_dim, learning_rate):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.learning_rate = learning_rate
            self.model = self.create_model()
            self.opt = tf.keras.optimizers.Adam(self.learning_rate)

    # 创建模型
    def create_model(self):
        state_input = Input((self.state_dim,))
        s1 = Dense(HIDDEN1_UNITS, activation='relu')(state_input)
        s2 = Dense(HIDDEN2_UNITS, activation='relu')(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(HIDDEN1_UNITS, activation='relu')(action_input)
        a2 = Dense(HIDDEN2_UNITS, activation='relu')(a1)
        c1 = concatenate([s2, a2], axis=-1)  # 将a2拼接在s2之后
        c2 = Dense(HIDDEN3_UNITS, activation='relu')(c1)
        output = Dense(1, activation='linear')(c2)
        return tf.keras.Model([state_input, action_input], output)

    # 返回预测值
    def predict(self, inputs):
        return self.model.predict(inputs)

    # 计算q值的梯度
    def q_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values, axis=-1)
        return tape.gradient(q_values, actions)

    # 计算loss
    def compute_loss(self, q_pred, q_target):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(q_target, q_pred)

    # 训练网络
    def train(self, states, actions, q_target):
        with tf.GradientTape() as tape:
            q_pred = self.model([states, actions], training=True)
            loss = self.compute_loss(q_pred, tf.stop_gradient(q_target))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss
