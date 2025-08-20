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
8-18
少数据量+无action 有效

8-19
不用action的loss，loss很快降到-1，可能是数据太少了（任务太单一了）

8-20
数据量非常大的时候，两种方法没区别

"""