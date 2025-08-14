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
from abc import ABC, abstractmethod

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
    output_dim: int = 2
    channel_dim: int = 1

    output_dir: str = "outputs/supervised"


device = get_device()


class TrainingState:
    """训练状态封装类"""

    def __init__(self):
        # 训练累积统计
        self.train_loss_acc = 0.0
        self.train_correct_acc = 0
        self.train_total_acc = 0
        self.epoch_step_count = 0

        # 训练进度
        self.current_step = 0
        self.current_epoch = 0

        # 最佳模型记录
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    def reset_epoch_stats(self):
        """重置epoch级别的累积统计"""
        self.train_loss_acc = 0.0
        self.train_correct_acc = 0
        self.train_total_acc = 0
        self.epoch_step_count = 0

    def update_train_stats(self, loss, correct, total):
        """更新训练统计信息"""
        self.train_loss_acc += loss
        self.train_correct_acc += correct
        self.train_total_acc += total
        self.epoch_step_count += 1

    def get_avg_train_loss(self):
        """获取平均训练损失"""
        return (
            self.train_loss_acc / self.epoch_step_count
            if self.epoch_step_count > 0
            else 0.0
        )

    def get_train_accuracy(self):
        """获取训练准确率"""
        return (
            100.0 * self.train_correct_acc / self.train_total_acc
            if self.train_total_acc > 0
            else 0.0
        )

    def update_best_model(self, val_acc, val_loss):
        """更新最佳模型记录"""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            return True
        return False


class FlappyBirdDataset(Dataset):
    """FlappyBird数据集"""

    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BaseModel(ABC):
    """模型训练细节抽象基类"""

    def __init__(self, config, device):
        self.config = config
        self.device = device

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets):
        pass

    def backward(self, loss):
        loss.backward()

    def parameters(self):
        # 需要子类实现
        raise NotImplementedError

    def train(self, mode=True):
        # 需要子类实现
        raise NotImplementedError

    def eval(self):
        # 需要子类实现
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        # 需要子类实现
        raise NotImplementedError

    def state_dict(self):
        # 需要子类实现
        raise NotImplementedError


class ThinkingModel(BaseModel):
    """具体模型实现，封装了Thinking网络"""

    def __init__(self, config, device):
        super().__init__(config, device)
        self.model = Thinking(config).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def backward(self, loss):
        loss.backward()

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()


class SupervisedTrainer:
    """FlappyBird 监督学习训练器"""

    def __init__(self, config: Config, model_class=None):
        self.config = config
        self.device = get_device()
        self.training_state = TrainingState()
        self.model_class = model_class or ThinkingModel

        # 设置输出目录
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        self.config.output_dir = os.path.join(config.output_dir, f"train_{timestamp}")
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 设置日志
        setup_logging(self.config.output_dir)

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.writer = None
        self.train_loader = None
        self.val_loader = None

    def save_model_checkpoint(self, checkpoint_type="checkpoint"):
        """保存模型检查点的辅助函数"""
        step = self.training_state.current_step
        epoch = self.training_state.current_epoch
        val_acc = self.training_state.best_val_acc
        val_loss = self.training_state.best_val_loss

        if checkpoint_type == "best":
            logging.info(f"保存最佳模型 (Step {step})")
            filename = f"best_model_{step}.pth"
            # delete last best model
            last_best_model = os.path.join(
                self.config.output_dir, f"best_model_{step - self.config.save_freq}.pth"
            )
            if os.path.exists(last_best_model):
                os.remove(last_best_model)
        elif checkpoint_type == "final":
            logging.info(f"最终模型 (Step {step})")
            filename = f"final_model_{step}.pth"
        else:
            logging.info(f"保存检查点 (Step {step})")
            filename = f"checkpoint_{step}.pth"

        model_path = os.path.join(self.config.output_dir, filename)

        metadata = {
            "epoch": epoch,
            "step": step,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }

        save_model(self.model, model_path, asdict(self.config), metadata)
        return model_path

    def preprocess_observations(self, observations, channel_dim=1):
        """预处理观测数据"""
        # 数据预处理 - 调整观测数据维度为 (N, C, H, W)
        if len(observations.shape) == 3:  # (N, H, W)
            observations = observations[:, None, :, :]  # 添加通道维度
        elif (
            len(observations.shape) == 4 and observations.shape[1] != 4
        ):  # (N, H, W, C)
            observations = observations.transpose(0, 3, 1, 2)  # 转换为 (N, C, H, W)

        # 如果观测数据只有1个通道，复制为4个通道以匹配模型输入
        if observations.shape[1] == 1:
            observations = np.repeat(observations, 4, axis=1)

        if channel_dim == 1:
            observations = observations[:, 3:4, :, :]  # 保留第4个通道

        return observations

    def evaluate_model_accuracy(self, model, data_loader):
        """评估模型准确率和损失"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for observations_batch, actions_batch in data_loader:
                observations_batch = observations_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)

                outputs = model.forward(observations_batch)
                loss = model.compute_loss(outputs, actions_batch)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += actions_batch.size(0)
                correct += (predicted == actions_batch).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def log_training_metrics(self, train_loss, train_acc, val_loss=None, val_acc=None):
        """记录训练指标到TensorBoard"""
        step = self.training_state.current_step
        epoch = self.training_state.current_epoch

        self.writer.add_scalar("Step/Train_Loss", train_loss, step)
        self.writer.add_scalar("Step/Train_Accuracy", train_acc, step)
        self.writer.add_scalar("Step/Epoch", epoch, step)

        if val_loss is not None and val_acc is not None:
            self.writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
            self.writer.add_scalar("Epoch/Train_Accuracy", train_acc, epoch)
            self.writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
            self.writer.add_scalar("Epoch/Val_Accuracy", val_acc, epoch)

    def load_and_preprocess_dataset(self):
        """加载和预处理数据集"""
        # 加载数据
        data = load_data(self.config.data_path)
        logging.info(f"data len: {len(data['observations'])}")

        data_size = (
            self.config.data_size
            if self.config.data_size > 0
            else len(data["observations"])
        )
        observations = data["observations"][:data_size]
        actions = data["actions"][:data_size]

        logging.info(f"观测数据形状: {observations.shape}")
        logging.info(f"动作数据形状: {actions.shape}")
        logging.info(f"数据样本数量: {len(observations)}")

        # 使用预处理函数
        observations = self.preprocess_observations(
            observations, self.config.channel_dim
        )
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

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

    def setup_training(self):
        """设置训练组件"""
        # 创建模型
        self.model = self.model_class(asdict(self.config), self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # TensorBoard记录器
        self.writer = SummaryWriter(self.config.output_dir)
        logging.info(f"TensorBoard日志保存在: {self.config.output_dir}")

    def _log_training_info(self):
        """记录训练配置信息"""
        logging.info("=" * 60)
        logging.info("FlappyBird 监督学习训练")
        logging.info("=" * 60)
        logging.info(f"配置:")
        for key, value in asdict(self.config).items():
            logging.info(f"  {key}: {value}")
        logging.info("=" * 60)
        logging.info(f"加载数据: {self.config.data_path}")

    def _perform_validation_and_save(self):
        """执行验证并保存模型"""
        # 计算训练平均值
        avg_train_loss = self.training_state.get_avg_train_loss()
        train_acc = self.training_state.get_train_accuracy()

        # 验证
        avg_val_loss, val_acc = self.evaluate_model_accuracy(
            self.model, self.val_loader
        )

        # 记录epoch级别的指标到TensorBoard
        self.log_training_metrics(avg_train_loss, train_acc, avg_val_loss, val_acc)

        step = self.training_state.current_step
        epoch = self.training_state.current_epoch

        logging.info(f"Epoch {epoch} 完成 (Step {step})")
        logging.info(f"训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        logging.info(f"验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        logging.info("-" * 60)

        # 保存检查点
        self.save_model_checkpoint(checkpoint_type="checkpoint")

        # 保存最佳模型
        if self.training_state.update_best_model(val_acc, avg_val_loss):
            self.save_model_checkpoint(checkpoint_type="best")

        return avg_val_loss, val_acc

    def train(self):
        """训练模型"""
        self._log_training_info()

        # 加载数据和设置训练
        self.load_and_preprocess_dataset()
        self.setup_training()

        steps_per_epoch = len(self.train_loader)
        logging.info(
            f"开始训练(总步数: {self.config.num_step}, 每轮步数: {steps_per_epoch})..."
        )

        # 创建数据加载器的无限迭代器
        train_iter = itertools.cycle(self.train_loader)

        for step in range(1, self.config.num_step + 1):
            self.training_state.current_step = step
            self.training_state.current_epoch = (step - 1) // steps_per_epoch + 1

            # 训练一步
            self.model.train()
            observations_batch, actions_batch = next(train_iter)
            observations_batch = observations_batch.to(self.device)
            actions_batch = actions_batch.to(self.device)

            # 前向和反向传播
            self.optimizer.zero_grad()
            outputs = self.model.forward(observations_batch)

            loss = self.model.compute_loss(outputs, actions_batch)
            self.model.backward(loss)

            self.optimizer.step()

            # 更新统计信息
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == actions_batch).sum().item()
            total = actions_batch.size(0)
            self.training_state.update_train_stats(loss.item(), correct, total)

            # 记录到TensorBoard
            self.log_training_metrics(
                loss.item(), self.training_state.get_train_accuracy()
            )

            # 打印训练进度
            if step % self.config.logging_freq == 0:
                logging.info(
                    f"Step [{step}/{self.config.num_step}] (Epoch {self.training_state.current_epoch}), "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {self.training_state.get_train_accuracy():.2f}%"
                )

            # 验证和保存
            if step % self.config.save_freq == 0:
                avg_val_loss, val_acc = self._perform_validation_and_save()
                # 重置累积统计
                self.training_state.reset_epoch_stats()

        # 保存最终模型
        avg_val_loss, val_acc = self.evaluate_model_accuracy(
            self.model, self.val_loader
        )
        self.save_model_checkpoint(checkpoint_type="final")

        self.writer.close()

        logging.info(f"\n训练完成！")
        logging.info(f"最佳验证准确率: {self.training_state.best_val_acc:.2f}%")
        logging.info(f"TensorBoard日志: {self.config.output_dir}")
        logging.info("运行以下命令查看训练过程:")
        logging.info(f"tensorboard --logdir {self.config.output_dir}")

    def evaluate_model(self, model_path: str):
        """评估保存的模型"""
        logging.info(f"加载模型: {model_path}")

        # 加载数据
        data = load_data(self.config.data_path)
        observations = data["observations"]
        actions = data["actions"]

        # 预处理
        observations = self.preprocess_observations(
            observations, self.config.channel_dim
        )

        # 创建测试数据集
        dataset = FlappyBirdDataset(observations, actions)
        test_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # 加载模型
        model = self.model_class(asdict(self.config)).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 评估
        avg_loss, accuracy = self.evaluate_model_accuracy(model, test_loader)

        # 统计动作分布
        action_counts = {0: 0, 1: 0}
        model.eval()
        with torch.no_grad():
            for observations_batch, _ in test_loader:
                observations_batch = observations_batch.to(self.device)
                outputs = model.forward(observations_batch)
                action_logits = outputs[:, -self.config.action_dim :]
                _, predicted = torch.max(action_logits, 1)

                for pred in predicted:
                    action_idx = int(pred.item())
                    action_counts[action_idx] += 1

        total = sum(action_counts.values())
        logging.info(f"测试准确率: {accuracy:.2f}%")
        logging.info(
            f"动作0预测数量: {action_counts[0]} ({100.0*action_counts[0]/total:.1f}%)"
        )
        logging.info(
            f"动作1预测数量: {action_counts[1]} ({100.0*action_counts[1]/total:.1f}%)"
        )

        return accuracy


# Remove the old standalone functions and update the train function
def train(config: Config):
    """训练模型"""
    trainer = SupervisedTrainer(config)
    trainer.train()


def evaluate_model(model_path: str, config: Config):
    """评估保存的模型"""
    trainer = SupervisedTrainer(config)
    return trainer.evaluate_model(model_path)


if __name__ == "__main__":
    # 使用tyro解析命令行参数
    config = tyro.cli(Config)

    train(config)
