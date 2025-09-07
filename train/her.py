import os
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# -------------------------
# 配置
# -------------------------
ENV_ID = "AdroitHandDoorSparse-v1"  # 你要训练的 env
TIMESTEPS = 2_000_000  # 根据算力和需求自行调整
EVAL_FREQ = 50_000  # 每多少 step 评估一次
N_EVAL_EPISODES = 10
SAVE_DIR = "./runs/her_sac_adroitdoor"
os.makedirs(SAVE_DIR, exist_ok=True)

# HER 参数
n_sampled_goal = 4
goal_selection_strategy = "future"  # 'future' is commonly used
online_sampling = True

# SAC 超参（可根据需要调）
sac_kwargs = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.98,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    verbose=1,
)


# -------------------------
# 环境包装函数
# -------------------------
def make_env(env_id):
    def _init():
        env = gym.make(env_id)
        # Monitor 用于记录 episode reward 等日志，便于 EvalCallback 与 TensorBoard 可视化
        env = Monitor(env)
        return env

    return _init


# 使用 DummyVecEnv（很多 SB3 算法期望向量化 env）
vec_env = DummyVecEnv([make_env(ENV_ID)])
# 可选：对 observation/reward 做标准化（对稀疏 reward 场景，有时不希望对 reward 做 Norm）
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# -------------------------
# 配置 HER replay buffer
# -------------------------
# HerReplayBuffer 需要和 env 一起工作，当使用 VecEnv 时 SB3 会自动把环境封装信息传递给 replay buffer
replay_buffer_class = HerReplayBuffer
replay_buffer_kwargs = dict(
    n_sampled_goal=n_sampled_goal,
    goal_selection_strategy=GoalSelectionStrategy.FUTURE,  # 或者使用字符串 'future'
    online_sampling=online_sampling,
    max_episode_length=100,  # 估计的 episode 长度，若不确定可用 env._max_episode_steps
)

# -------------------------
# 创建模型（使用 MultiInputPolicy 以处理 dict observation）
# -------------------------
model = SAC(
    policy="MultiInputPolicy",
    env=vec_env,
    replay_buffer_class=replay_buffer_class,
    replay_buffer_kwargs=replay_buffer_kwargs,
    **sac_kwargs,
)

# -------------------------
# 评估回调（评估时使用未标准化的环境副本）
# -------------------------
# 为评估创建单独的 env（不要用 VecNormalize 的 env，否则需要反标准化）
eval_env = make_env(ENV_ID)()
# 如果你需要对评估使用 VecNormalize 的反归一化，需要单独处理，这里做最简单的用法：不做标准化的 eval env
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=SAVE_DIR,
    log_path=SAVE_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
)

# Checkpoints：定期保存模型（可选）
checkpoint_callback = CheckpointCallback(
    save_freq=EVAL_FREQ, save_path=SAVE_DIR, name_prefix="her_sac_checkpoint"
)

# -------------------------
# 开始训练
# -------------------------
print("Start training SAC+HER on", ENV_ID)
model.learn(total_timesteps=TIMESTEPS, callback=[eval_callback, checkpoint_callback])

# 保存最终模型与 VecNormalize（如果使用）
model.save(os.path.join(SAVE_DIR, "her_sac_adroitdoor_final"))
vec_norm_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")
vec_env.save(vec_norm_path)
print("Saved model and VecNormalize to", SAVE_DIR)

# -------------------------
# 简单评估：加载模型并评估成功率（使用评估环境）
# -------------------------
# 注意：若你保存了 VecNormalize，需要加载并反归一化 obs/rewards（此处为了简单直接评估不使用 VecNormalize）
# load_model = SAC.load(os.path.join(SAVE_DIR, "her_sac_adroitdoor_final"))
# load_model.set_env(make_env(ENV_ID)())  # or set vec env properly if you used VecNormalize

# 或直接使用当前 model
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=20,
    deterministic=True,
)
print(f"Evaluation mean_reward: {mean_reward:.3f} +/- {std_reward:.3f}")
