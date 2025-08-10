import itertools
import logging
import numpy as np
import tyro
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from dataclasses import asdict, dataclass

import sys

sys.path.append(os.getcwd())

from src.utils import get_device, save_model, setup_logging
from src.dataset import load_data
from src.net.thinking import Thinking


@dataclass
class Config:
    # 数据路径
    data_path: str = (
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250806_003553.h5"
    )
    data_size: int = 20000

    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_step: int = 4000
    save_freq: int = num_step // 4
    logging_freq: int = 200

    # 模型参数
    action_dim: int = 2
    feal_dim: int = 0
    channel_dim: int = 1

    output_dir: str = "outputs/supervised"


device = get_device()


class FlappyBirdDataset(Dataset):
    """FlappyBird数据集"""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def save_model_checkpoint(
    model,
    config,
    output_dir,
    epoch,
    step,
    val_acc,
    val_loss,
    checkpoint_type="checkpoint",
):
    """保存模型检查点的辅助函数"""
    if checkpoint_type == "best":
        logging.info(f"保存最佳模型 (Step {step})")
        filename = f"best_model_{step}.pth"
        # delete last best model
        last_best_model = os.path.join(
            output_dir, f"best_model_{step - config.save_freq}.pth"
        )
        if os.path.exists(last_best_model):
            os.remove(last_best_model)
    elif checkpoint_type == "final":
        logging.info(f"最终模型 (Step {step})")
        filename = f"final_model_{step}.pth"
    else:
        logging.info(f"保存检查点 (Step {step})")
        filename = f"checkpoint_{step}.pth"

    model_path = os.path.join(output_dir, filename)

    metadata = {
        "epoch": epoch,
        "step": step,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }

    save_model(model, model_path, asdict(config), metadata)

    return model_path


def load_and_preprocess_dataset(config):

    # 加载数据
    data = load_data(config.data_path)
    logging.info(f"data len: {len(data['observations'])}")

    data_size = config.data_size if config.data_size > 0 else len(data["observations"])
    observations = data["observations"][:data_size]
    actions = data["actions"][:data_size]

    logging.info(f"观测数据形状: {observations.shape}")
    logging.info(f"动作数据形状: {actions.shape}")
    logging.info(f"数据样本数量: {len(observations)}")

    # 数据预处理 - 调整观测数据维度为 (N, C, H, W)
    if len(observations.shape) == 3:  # (N, H, W)
        observations = observations[:, None, :, :]  # 添加通道维度
    elif len(observations.shape) == 4 and observations.shape[1] != 4:  # (N, H, W, C)
        observations = observations.transpose(0, 3, 1, 2)  # 转换为 (N, C, H, W)

    # 如果观测数据只有1个通道，复制为4个通道以匹配模型输入
    if observations.shape[1] == 1:
        observations = np.repeat(observations, 4, axis=1)

    if config.channel_dim == 1:
        observations = observations[:, 3:4, :, :]  # 保留第4个通道

    logging.info(f"预处理后观测数据形状: {observations.shape}")

    # 创建数据集和数据加载器
    dataset = FlappyBirdDataset(observations, actions)

    # 划分训练和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader


def train(config: Config):
    """训练模型"""

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"train_{timestamp}")

    # 创建保存目录
    os.makedirs(config.output_dir, exist_ok=True)
    setup_logging(config.output_dir)

    logging.info("=" * 60)
    logging.info("FlappyBird 监督学习训练")
    logging.info("=" * 60)
    logging.info(f"配置:")
    for key, value in asdict(config).items():
        logging.info(f"  {key}: {value}")
    logging.info("=" * 60)

    logging.info(f"加载数据: {config.data_path}")

    train_loader, val_loader = load_and_preprocess_dataset(config)

    # 创建模型
    model = Thinking(asdict(config))
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard记录器
    writer = SummaryWriter(config.output_dir)
    logging.info(f"TensorBoard日志保存在: {config.output_dir}")

    steps_per_epoch = len(train_loader)
    logging.info(f"开始训练(总步数: {config.num_step}, 每轮步数: {steps_per_epoch})...")

    # 创建数据加载器的无限迭代器
    train_iter = itertools.cycle(train_loader)

    # 训练和验证累积统计
    best_val_acc = 0.0
    train_loss_acc = 0.0
    train_correct_acc = 0
    train_total_acc = 0
    epoch_step_count = 0

    for step in range(1, config.num_step + 1):
        model.train()

        observations_batch, actions_batch = next(train_iter)

        observations_batch = observations_batch.to(device)
        actions_batch = actions_batch.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(observations_batch)
        action_logits = outputs[:, -config.action_dim :]
        loss = criterion(action_logits, actions_batch)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累积统计
        train_loss_acc += loss.item()
        _, predicted = torch.max(action_logits.data, 1)
        train_total_acc += actions_batch.size(0)
        train_correct_acc += (predicted == actions_batch).sum().item()
        epoch_step_count += 1

        # 计算当前epoch
        current_epoch = (step - 1) // steps_per_epoch + 1

        # 记录到TensorBoard
        writer.add_scalar("Step/Train_Loss", loss.item(), step)
        writer.add_scalar(
            "Step/Train_Accuracy", 100.0 * train_correct_acc / train_total_acc, step
        )
        writer.add_scalar("Step/Epoch", current_epoch, step)

        # 打印训练进度
        if step % config.logging_freq == 0:
            logging.info(
                f"Step [{step}/{config.num_step}] (Epoch {current_epoch}), "
                f"Loss: {loss.item():.4f}, "
                f"Acc: {100.0 * train_correct_acc / train_total_acc:.2f}%"
            )

        if step % config.save_freq == 0:
            # 计算训练平均值
            avg_train_loss = train_loss_acc / epoch_step_count
            train_acc = 100.0 * train_correct_acc / train_total_acc

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for observations_batch, actions_batch in val_loader:
                    observations_batch = observations_batch.to(device)
                    actions_batch = actions_batch.to(device)

                    outputs = model(observations_batch)
                    action_logits = outputs[:, -config.action_dim :]

                    loss = criterion(action_logits, actions_batch)
                    val_loss += loss.item()

                    _, predicted = torch.max(action_logits.data, 1)
                    val_total += actions_batch.size(0)
                    val_correct += (predicted == actions_batch).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # 记录epoch级别的指标到TensorBoard
            writer.add_scalar("Epoch/Train_Loss", avg_train_loss, current_epoch)
            writer.add_scalar("Epoch/Train_Accuracy", train_acc, current_epoch)
            writer.add_scalar("Epoch/Val_Loss", avg_val_loss, current_epoch)
            writer.add_scalar("Epoch/Val_Accuracy", val_acc, current_epoch)

            logging.info(f"Epoch {current_epoch} 完成 (Step {step})")
            logging.info(f"训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
            logging.info(f"验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
            logging.info("-" * 60)

            save_model_checkpoint(
                model,
                config,
                config.output_dir,
                current_epoch,
                step,
                val_acc,
                avg_val_loss,
                checkpoint_type="checkpoint",
            )

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model_checkpoint(
                    model,
                    config,
                    config.output_dir,
                    current_epoch,
                    step,
                    val_acc,
                    avg_val_loss,
                    checkpoint_type="best",
                )

            # 重置累积统计
            train_loss_acc = 0.0
            train_correct_acc = 0
            train_total_acc = 0
            epoch_step_count = 0

    # 保存最终模型
    final_epoch = (config.num_step - 1) // steps_per_epoch + 1
    save_model_checkpoint(
        model,
        config,
        config.output_dir,
        final_epoch,
        config.num_step,
        val_acc,
        avg_val_loss,
        checkpoint_type="final",
    )

    writer.close()

    logging.info(f"\n训练完成！")
    logging.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    logging.info(f"TensorBoard日志: {config.output_dir}")
    logging.info("运行以下命令查看训练过程:")
    logging.info(f"tensorboard --logdir {config.output_dir}")


def evaluate_model(model_path: str, config: Config):
    """评估保存的模型"""
    logging.info(f"加载模型: {model_path}")

    # 加载数据
    data = load_data(config.data_path)
    observations = data["observations"]
    actions = data["actions"]

    # 预处理数据
    if len(observations.shape) == 3:
        observations = observations[:, None, :, :]
    elif len(observations.shape) == 4 and observations.shape[1] != 4:
        observations = observations.transpose(0, 3, 1, 2)

    if observations.shape[1] == 1:
        observations = np.repeat(observations, 4, axis=1)

    # 创建测试数据集
    dataset = FlappyBirdDataset(observations, actions)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # 加载模型
    model = Thinking(config.feature_dim, config.action_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 评估
    correct = 0
    total = 0
    action_counts = {0: 0, 1: 0}  # 统计每个动作的预测数量

    with torch.no_grad():
        for observations_batch, actions_batch in test_loader:
            observations_batch = observations_batch.to(device)
            actions_batch = actions_batch.to(device)

            outputs = model(observations_batch)
            action_logits = outputs[:, -config.action_dim :]

            _, predicted = torch.max(action_logits, 1)
            total += actions_batch.size(0)
            correct += (predicted == actions_batch).sum().item()

            # 统计预测的动作分布
            for pred in predicted:
                action_idx = int(pred.item())
                action_counts[action_idx] += 1

    accuracy = 100.0 * correct / total
    logging.info(f"测试准确率: {accuracy:.2f}%")
    logging.info(
        f"动作0预测数量: {action_counts[0]} ({100.0*action_counts[0]/total:.1f}%)"
    )
    logging.info(
        f"动作1预测数量: {action_counts[1]} ({100.0*action_counts[1]/total:.1f}%)"
    )

    return accuracy


if __name__ == "__main__":
    # 使用tyro解析命令行参数
    config = tyro.cli(Config)

    train(config)
