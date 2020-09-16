# 导入相关模块
import numpy as np
import gym
import tensorflow as tf
from SDN import Actor, Critic, OUNoise, ReplayBuffer
from matplotlib import pyplot as plt

# 软更新target网络
def target_update(actor, actor_target, critic, critic_target, tau):
    actor_weights = actor.model.get_weights()
    t_actor_weights = actor_target.model.get_weights()
    critic_weights = critic.model.get_weights()
    t_critic_weights = critic_target.model.get_weights()

    for i in range(len(actor_weights)):
        t_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * t_actor_weights[i]

    for i in range(len(critic_weights)):
        t_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * t_critic_weights[i]

    actor_target.model.set_weights(t_actor_weights)
    critic_target.model.set_weights(t_critic_weights)

# 计算yi
def compute_yi(rewards, target_q_values, dones, GAMMA):
    yi = np.asarray(target_q_values)
    for i in range(target_q_values.shape[0]):
        if dones[i]:
            yi[i] = rewards[i]
        else:
            yi[i] = GAMMA * target_q_values[i] + rewards[i]
    return yi


def play(env):
    # 相关常量的定义
    BUFFER_SIZE = 20000  # 缓冲池的大小
    BATCH_SIZE = 64  # batch_size的大小
    GAMMA = 0.99  # 折扣系数
    TAU = 0.05  # target网络软更新的速度
    LR_A = 0.0005 # Actor网络的学习率
    LR_C = 0.001  # Critic网络的学习率

    # 相关变量的定义
    episode = 1000  # 迭代的次数
    explore = 800  # 每次需要与环境交互的步数
    total_step = 0  # 总共运行了多少步
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # 可视化集合定义
    reward_list = []  # 记录所有的rewards进行可视化展示
    loss_list = []  # 记录损失函数进行可视化展示
    step_list = []  # 记录每一步的结果
    ep_list = []
    step_list_count = 0  # 绘图下标

    # 神经网络相关操作定义
    OU = OUNoise.OU()  # 引入噪声
    buff = ReplayBuffer.ReplayBuffer(BUFFER_SIZE)   # 创建缓冲区

    # 创建四个神经网络
    actor = Actor.Actor(state_dim, action_dim, action_bound, LR_A)
    actor_target = Actor.Actor(state_dim, action_dim, action_bound, LR_A)
    critic = Critic.Critic(state_dim, action_dim, LR_C)
    critic_target = Critic.Critic(state_dim, action_dim, LR_C)

    # 给target网络设置参数
    actor_weight = actor.model.get_weights()
    critic_weight = critic.model.get_weights()
    actor_target.model.set_weights(actor_weight)
    critic_target.model.set_weights(critic_weight)

    # 加载训练数据
    # print("Now we load the weight")
    # try:
    #     actor.model.load_weights("src/actormodel.h5")
    #     critic.model.load_weights("src/criticmodel.h5")
    #     actor_target.model.load_weights("src/actormodel.h5")
    #     critic_target.model.load_weights("src/criticmodel.h5")
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    # 开始迭代
    print("Experiment Start.")
    for ep in range(episode):

        # 输出当前信息
        print("Episode : " + str(ep) + " Replay Buffer " + str(buff.getCount()))

        total_reward = 0
        total_loss = 0

        state = env.reset()
        state = state.reshape(state_dim, -1)
        done = False
        noise = np.zeros(action_dim)
        step = 0
        while not done:
            env.render()
            state = state.T
            action = actor.model.predict(state)
            if ep <= explore:
                noise = OU.function(noise, dim=action_dim)
                action = np.clip(action + noise, -action_bound, action_bound)
            next_state, reward, done, _ = env.step(action)
            buff.add(state.squeeze(), action, reward, next_state, done)

            # 取样进行更新

            states, actions, rewards, next_states, dones = buff.getBatch(BATCH_SIZE)

            target_q_values = critic_target.model.predict([next_states, actor_target.predict(next_states)])
            yi = compute_yi(rewards, target_q_values, dones, GAMMA)
            loss = critic.train(states, actions, yi)    # 计算损失函数，并更新网络参数

            a_for_grads = actor.predict(states)
            a_grads = critic.q_grads(states, a_for_grads)
            actor.train(states, a_grads)
            target_update(actor, actor_target, critic, critic_target, TAU)  # 网络更新

            print("Episode", ep, "Step", step, "Action", action, "Reward", reward, "Loss", np.array(loss))

            total_reward += reward
            total_loss += np.array(loss)
            step += 1
            total_step += 1
            state = next_state

        # 绘制图像，并保存
        reward_list.append(total_reward)
        loss_list.append(total_loss / step)
        ep_list.append(ep)

        # 保存参数模型
        print("Now we save model")
        # actor.model.save("src/actormodel.h5", overwrite=True)
        # critic.model.save("src/criticmodel.h5", overwrite=True)

        # 打印相关信息
        print("")
        print("-" * 50)
        print("TOTAL REWARD @ " + str(ep) + "-th Episode  : Reward " + str(total_reward))
        print("TOTAL LOSS @ " + str(ep) + "-th Episode  : LOSS " + str(total_loss / step))
        print("Total Step: " + str(total_step))
        print("-" * 50)
        print("")

    plt.cla()
    plt.plot(ep_list, reward_list)
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("reward-step")
    img_name = "img/reward/" + "reward"
    plt.savefig(img_name)

    plt.cla()  # 清除
    plt.plot(ep_list, loss_list)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("loss-step")
    img_name = "img/loss/" + "loss"
    plt.savefig(img_name)

if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    env.reset()
    env.render()
    play(env)
