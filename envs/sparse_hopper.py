import gymnasium as gym


class SparseHopper(gym.Wrapper):
    def __init__(self, env, goal_distance=1.0):
        super().__init__(env)
        self.goal_distance = goal_distance

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        distance = info["x_position"]

        # 定义稀疏奖励
        sparse_reward = 0.0
        # 如果成功到达目标距离
        if distance >= self.goal_distance:
            sparse_reward = 1.0
            terminated = True  # 成功也终止回合

        # 完全忽略原本的密集奖励
        return observation, sparse_reward, terminated, truncated, info
