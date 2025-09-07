import os
import gymnasium as gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor


def create_env(env_id, eval_env=False):
    """
    创建环境并包装为VecEnv
    """
    # 创建单个环境并用Monitor包装以记录日志
    env = make_vec_env(
        env_id,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        wrapper_class=Monitor,  # Monitor用于记录episode的return和length
        env_kwargs={"max_episode_steps": 200},
    )  # 可以设置最大步数

    # 注意：对于Adroit环境，通常不建议使用VecNormalize进行观察值标准化，
    # 因为其状态空间非常复杂且包含不同物理量纲。但我们可以尝试标准化奖励。
    # 如果训练不稳定，可以注释掉VecNormalize这行。
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    return env


def main():
    # 环境ID
    env_id = "AdroitHandDoorSparse-v1"  # 稀疏奖励版本

    # 创建训练环境和评估环境
    train_env = create_env(env_id)
    eval_env = create_env(env_id, eval_env=True)

    # 设置保存日志和模型的路径
    log_dir = "./her_sac_door_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 初始化SAC模型，并整合HER replay buffer
    model = SAC(
        policy="MultiInputPolicy",  # 必须使用MultiInputPolicy，因为环境观测是字典类型
        env=train_env,
        learning_rate=3e-4,  # 学习率
        buffer_size=1_000_000,  # 经验回放缓冲区总大小
        batch_size=1024,  # 每次梯度更新使用的batch size
        tau=0.005,  # 目标网络更新系数
        gamma=0.98,  # 折扣因子
        ent_coef="auto",  # 自动调整熵系数
        verbose=1,  # 输出训练信息
        tensorboard_log=log_dir,  # Tensorboard日志目录
        device="auto",  # 自动选择GPU或CPU
        replay_buffer_class=HerReplayBuffer,  # 使用HER回放缓冲区
        # HER的特定参数
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # 对于每个transition，采样4个新目标
            goal_selection_strategy="future",  # 从未来的状态中选取新目标
        ),
    )

    # 创建回调函数
    # 评估回调：定期评估模型性能
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir + "eval_logs/",
        eval_freq=5000,  # 每5000步评估一次
        deterministic=True,  # 评估时使用确定性动作
        render=False,  # 评估时不渲染（渲染会大大减慢速度）
        n_eval_episodes=5,  # 每次评估运行5个episode
    )

    # 检查点回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 每10000步保存一次
        save_path=log_dir + "checkpoints/",
        name_prefix="her_sac_door",
    )

    # 将回调组合成一个列表
    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # 开始训练！
    print("开始训练 SAC + HER on AdroitHandDoorSparse...")
    print(f"日志和模型将保存在: {log_dir}")
    print("可以在终端运行: tensorboard --logdir=./her_sac_door_logs/ 来查看训练进度")

    model.learn(
        total_timesteps=2_000_000,  # 总训练步数，对于此任务可能需要更多
        callback=callback_list,  # 传入回调函数
        tb_log_name="SAC_HER_run",  # Tensorboard中的运行名称
    )

    # 训练结束后，保存最终模型
    model.save(log_dir + "her_sac_door_final_model")
    print("训练完成，最终模型已保存。")

    # 关闭环境
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
