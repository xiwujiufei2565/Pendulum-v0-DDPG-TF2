# 导入相关依赖包
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
import numpy

# 常量定义
HIDDEN1_UNITS = 50 # 第一层神经元
HIDDEN2_UNITS = 50 # 第二层神经元

# Actor类
class Actor:
    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)

    # 创建Actor模型
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(HIDDEN1_UNITS, activation='relu'),
            Dense(HIDDEN2_UNITS, activation='relu'),
            Dense(self.action_dim, activation='tanh'),
            Lambda(lambda x: x * self.action_bound)
        ])
        return model

    # 训练网络
    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.model(states), self.model.trainable_weights, -q_grads)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

    # 预测结果
    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]
