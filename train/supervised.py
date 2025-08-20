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
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250808_005810.h5"
    )
    eval_data_path: str = (
        "outputs/dataset_s4/observations_actions_flappy_bird_800000_20250806_003553.h5"
    )
    data_size: int = 30000

    # 训练参数
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_step: int = 8000
    save_freq: int = 400
    logging_freq: int = 200

    # 模型参数
    output_dim: int = 2
    channel_dim: int = 4

    name: str = "supervised"
    output_dir: str = "outputs/8-19/data_size_30000"


device = get_device()


class TrainingState:
    """训练状态封装类"""

    def __init__(self, config: Config):
        self.config = config

        self.steps_per_epoch = 0

    def start(self):
        self.writer = SummaryWriter(self.config.output_dir)
        logging.info(f"TensorBoard日志保存在: {self.config.output_dir}")

        # 训练进度
        self.current_step = 0
        self.current_epoch = 0

        # 最佳模型记录
        self.best_eval_acc = 0.0
        self.best_eval_loss = float("inf")

        """记录训练配置信息"""
        logging.info("=" * 60)
        logging.info(f"FlappyBird : {self.config.name}")
        logging.info("=" * 60)
        logging.info(f"配置:")
        for key, value in asdict(self.config).items():
            logging.info(f"  {key}: {value}")
        logging.info("=" * 60)

    def log_step(self, step, loss_dict, acc):
        """更新训练状态"""
        self.current_step = step
        self.current_epoch = (step - 1) // self.steps_per_epoch + 1

        self.writer.add_scalar("Train/Train_Accuracy", 100.0 * acc, step)
        self.writer.add_scalar("Train/Epoch", self.current_epoch, step)

        loss_log = ""
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Train/Train_{k}", v.item(), step)
            loss_log += f"{k}: {v.item():.4f}, "

        # 打印训练进度
        if step % self.config.logging_freq == 0:
            logging.info(
                f"Step [{step}/{self.config.num_step}] (Epoch {self.current_epoch}), "
                f"Acc: {100.0 * acc:.2f}%, "
                f"{loss_log}"
            )

    def update_best_model(self, val_acc, val_loss):
        """更新最佳模型记录"""
        if val_acc > self.best_eval_acc:
            self.best_eval_acc = val_acc
            self.best_eval_loss = val_loss
            return True
        return False

    def log_evaluation(self, loss_logs, acc):
        """记录评估指标到TensorBoard"""
        step = self.current_step
        logging.info(f"Evaluation: [{step}] Acc: {acc * 100:.4f}%")

        self.writer.add_scalar("Train/Eval_Accuracy", acc, step)

        loss_log = ""
        for k, v in loss_logs.items():
            self.writer.add_scalar(f"Train/Eval_{k}", v, step)
            loss_log += f"{k}: {v:.4f}, "

        logging.info(f"Evaluation: [{step}]" f"Acc: {acc * 100:.4f}%, " f"{loss_log}")

    def finish(self):

        self.writer.close()
        logging.info("=" * 60)
        logging.info(f"---- 训练完成！ ----")
        logging.info(f"最佳验证准确率: {self.best_eval_acc * 100:.4f}%")
        logging.info(f"TensorBoard日志: {self.config.output_dir}")
        logging.info("运行以下命令查看训练过程:")
        logging.info(f"tensorboard --logdir {self.config.output_dir}")


class FlappyBirdDataset(Dataset):
    """FlappyBird数据集"""

    def __init__(self, config, observations, actions):
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
    def forward(self, batch):
        pass

    @abstractmethod
    def compute_loss(self, outputs, batch):
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

    def forward(self, batch):
        x = batch[0]
        return (self.model(x),)

    def compute_loss(self, outputs, batch):
        targets = batch[1]
        ce_loss = self.criterion(outputs[0], targets)
        # 如有其他loss，可在此添加
        loss_dict = {
            "loss": ce_loss
            # "other_loss": other_loss  # 示例
        }
        return loss_dict

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

    def __init__(self, config: Config, model_class=None, dataset_class=None):
        self.config = config
        self.device = get_device()
        self.model_class = model_class or ThinkingModel
        self.dataset_class = dataset_class or FlappyBirdDataset

        # 设置输出目录
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        self.config.output_dir = os.path.join(
            config.output_dir, self.config.name, f"train_{self.config.name}_{timestamp}"
        )
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 设置日志
        setup_logging(self.config.output_dir)

        self.training_state = TrainingState(self.config)

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None

    def save_model_checkpoint(self, checkpoint_type="checkpoint"):
        """保存模型检查点的辅助函数"""
        step = self.training_state.current_step
        epoch = self.training_state.current_epoch
        eval_acc = self.training_state.best_eval_acc
        eval_loss = self.training_state.best_eval_loss

        if checkpoint_type == "best":
            logging.info(
                f"保存最佳模型 Step {step}, acc: {eval_acc * 100:.4f}%, loss: {eval_loss:.4f}"
            )
            filename = f"best_model.pth"
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
            "eval_acc": eval_acc,
            "eval_loss": eval_loss,
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

    def evaluate_model_accuracy(self):
        """评估模型准确率和损失"""
        data_loader = self.val_loader
        model = self.model
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_logs = {}

        with torch.no_grad():
            for batch in data_loader:
                for i in range(len(batch)):
                    batch[i] = batch[i].to(self.device)

                outputs = model.forward(batch)
                loss_dict = model.compute_loss(outputs, batch)
                loss = loss_dict["loss"]
                total_loss += loss.item()

                # 记录所有loss
                for k, v in loss_dict.items():
                    if k not in loss_logs:
                        loss_logs[k] = 0.0
                    loss_logs[k] += v.item()

                _, predicted = torch.max(outputs[0].data, 1)
                total += batch[-1].size(0)
                correct += (predicted == batch[-1]).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        # 计算所有loss的平均值
        avg_loss_logs = {k: v / len(data_loader) for k, v in loss_logs.items()}
        return avg_loss, accuracy, avg_loss_logs

    def prepare_training_loader(self):
        """加载和预处理数据集"""
        logging.info("******** 加载和预处理: 训练数据集 ********")
        # 加载数据
        data = load_data(self.config.data_path)
        logging.info(f"data len: {len(data['observations'])}")

        data_size = self.config.data_size

        assert data_size <= len(
            data["observations"]
        ), f"数据样本数量不足: {data_size} > {len(data['observations'])}"

        logging.info(f"使用数据样本数量: {data_size}")
        observations = data["observations"][:data_size]
        actions = data["actions"][:data_size]

        logging.info(f"观测数据形状: {observations.shape}")
        logging.info(f"动作数据形状: {actions.shape}")

        # 使用预处理函数
        observations = self.preprocess_observations(
            observations, self.config.channel_dim
        )
        logging.info(f"预处理后观测数据形状: {observations.shape}")

        # 创建数据集和数据加载器
        dataset = self.dataset_class(self.config, observations, actions)

        logging.info(f"训练集大小: {len(dataset)}")
        self.train_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def prepare_eval_loader(self):
        logging.info("******** 加载和预处理: 验证数据集 ********")
        data = load_data(self.config.eval_data_path)
        logging.info(f"eval data len: {len(data['observations'])}")
        observations = data["observations"][:30000]
        actions = data["actions"][:30000]

        # 使用预处理函数
        observations = self.preprocess_observations(
            observations, self.config.channel_dim
        )
        logging.info(f"预处理后观测数据形状: {observations.shape}")

        # 创建数据集和数据加载器
        dataset = self.dataset_class(self.config, observations, actions)

        logging.info(f"验证集大小: {len(dataset)}")

        self.val_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def setup_training(self):
        """设置训练组件"""
        # 创建模型
        self.model = self.model_class(asdict(self.config), self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def _perform_validation_and_save(self, checkpoint_type="checkpoint"):
        logging.info("--- Evaluate ---")

        # 验证
        avg_eval_loss, eval_acc, eval_loss_logs = self.evaluate_model_accuracy()

        self.training_state.log_evaluation(eval_loss_logs, eval_acc)

        # 保存检查点
        self.save_model_checkpoint(checkpoint_type)

        # 保存最佳模型
        if self.training_state.update_best_model(eval_acc, avg_eval_loss):
            self.save_model_checkpoint(checkpoint_type="best")

    def train(self):
        """训练模型"""
        self.training_state.start()

        # 加载数据和设置训练
        self.prepare_training_loader()
        self.prepare_eval_loader()
        self.setup_training()

        steps_per_epoch = len(self.train_loader)
        self.training_state.steps_per_epoch = steps_per_epoch
        logging.info("=" * 60)
        logging.info(
            f"---- 开始训练(总步数: {self.config.num_step}, 每轮步数: {steps_per_epoch}) ----"
        )

        # 创建数据加载器的无限迭代器
        train_iter = itertools.cycle(self.train_loader)

        for step in range(1, self.config.num_step + 1):

            # 训练一步
            self.model.train()
            batch = next(train_iter)

            """ Note: batch[-1] is label """
            for i in range(len(batch)):
                batch[i] = batch[i].to(self.device)

            # 前向和反向传播
            self.optimizer.zero_grad()
            outputs = self.model.forward(batch)

            loss_dict = self.model.compute_loss(outputs, batch)
            loss = loss_dict["loss"]
            self.model.backward(loss)

            self.optimizer.step()

            # 更新统计信息
            label = batch[-1]
            _, predicted = torch.max(outputs[0].data, 1)
            correct_num = (predicted == label).sum().item()
            acc = correct_num / label.size(0)

            self.training_state.log_step(step, loss_dict, acc)

            # 验证和保存
            if step % self.config.save_freq == 0:
                self._perform_validation_and_save()

        # 保存最终模型
        self._perform_validation_and_save("final")

        self.training_state.finish()

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
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 加载模型
        model = self.model_class(asdict(self.config)).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 评估
        avg_loss, accuracy = self.evaluate_model_accuracy()

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
