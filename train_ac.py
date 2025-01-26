import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time

from src.ac import ActorCritic
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

"""
可以运行的版本，但是貌似会陷入局部最优
"""
# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 300
max_test_ep = 400
image_size = 84


def train(global_model, rank):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = FlappyBird()

    for n_epi in range(max_train_ep):
        done = False

        image, reward, terminal = env.next_frame(0)
        image = pre_processing(
            image[: env.screen_width, : int(env.base_y)],
            image_size,
            image_size,
        )
        image = torch.from_numpy(image)

        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):

                prob = local_model.pi(state)[0]
                m = Categorical(prob)
                a = m.sample().item()
                image_prime, r, done = env.next_frame(a)
                image_prime = pre_processing(
                    image_prime[: env.screen_width, : int(env.base_y)],
                    image_size,
                    image_size,
                )
                image_prime = torch.from_numpy(image_prime)

                next_state = torch.cat((state[0, 1:, :, :], image_prime))[None, :, :, :]

                s_lst.append(state)
                a_lst.append([a])
                r_lst.append(r)

                state = next_state
                if done:
                    break

            R = 0.0 if done else local_model.v(next_state).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = (
                # torch.tensor(s_lst, dtype=torch.float),
                torch.cat(s_lst),
                torch.tensor(a_lst),
                torch.tensor(td_target_lst),
            )
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(
                local_model.v(s_batch), td_target.detach()
            )

            optimizer.zero_grad()
            loss.mean().backward()
            for global_param, local_param in zip(
                global_model.parameters(), local_model.parameters()
            ):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = FlappyBird()
    score = 0.0
    print_interval = 20

    image, reward, terminal = env.next_frame(0)
    image = pre_processing(
        image[: env.screen_width, : int(env.base_y)],
        image_size,
        image_size,
    )
    image = torch.from_numpy(image)

    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    for n_epi in range(max_test_ep):
        done = False

        while not done:
            prob = global_model.pi(state)[0]
            a = Categorical(prob).sample().item()
            image_prime, r, done = env.next_frame(a)

            image_prime = pre_processing(
                image_prime[: env.screen_width, : int(env.base_y)],
                image_size,
                image_size,
            )
            image_prime = torch.from_numpy(image_prime)

            next_state = torch.cat((state[0, 1:, :, :], image_prime))[None, :, :, :]
            state = next_state
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score / print_interval
                )
            )
            score = 0.0
            time.sleep(1)


if __name__ == "__main__":
    global_model = ActorCritic()
    global_model.share_memory()

    processes = []
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,))
        else:
            p = mp.Process(
                target=train,
                args=(
                    global_model,
                    rank,
                ),
            )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
