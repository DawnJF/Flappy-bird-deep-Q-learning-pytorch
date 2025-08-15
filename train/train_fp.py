"""
data_path: str = (
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250806_003553.h5"
    )
data_size: int = 1000
结论：学不会,坚持不了几个桩子

data_size: int = 10000
结论: best 能跑一会, final不行

data_size: int = 20000
结论: final 能跑, best 一小会

"""

"""
v1 版本：
两边都用 projector
只计算feal_loss 
loss直接 -1 
不可行

v2 版本：
x1用 projector
loss直接 -1 
不可行

"""


"""
JepaThinkModelV2
Evaluation Results:
Model: <class 'src.net.jepa_thinking.JepaThinking'>/outputs/compare/train_2025_0815_165647/checkpoint_1000.pth, Avg Steps: 18.0
Model: <class 'src.net.jepa_thinking.JepaThinking'>/outputs/compare/train_2025_0815_165647/checkpoint_2000.pth, Avg Steps: 264.4
Model: <class 'src.net.jepa_thinking.JepaThinking'>/outputs/compare/train_2025_0815_165647/checkpoint_3000.pth, Avg Steps: 626.3
Model: <class 'src.net.jepa_thinking.JepaThinking'>/outputs/compare/train_2025_0815_165647/checkpoint_4000.pth, Avg Steps: 825.2
Model: <class 'src.net.thinking.Thinking'>/outputs/compare/train_2025_0815_170228/checkpoint_1000.pth, Avg Steps: 18.0
Model: <class 'src.net.thinking.Thinking'>/outputs/compare/train_2025_0815_170228/checkpoint_2000.pth, Avg Steps: 309.8
Model: <class 'src.net.thinking.Thinking'>/outputs/compare/train_2025_0815_170228/checkpoint_3000.pth, Avg Steps: 345.2
Model: <class 'src.net.thinking.Thinking'>/outputs/compare/train_2025_0815_170228/checkpoint_4000.pth, Avg Steps: 230.9
"""
