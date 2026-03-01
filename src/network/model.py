import torch
import torch.nn.functional as F
import os
import shutil
from rich import print
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from .tdnn import ECAPA_TDNN
from .classifier import AAMClassifier
from .config import TDNNConfig, TrainConfig


class SimpleSV:
    """
    简单说话人识别模型
    """

    def __init__(self, config: TDNNConfig, device: torch.device = torch.device("cpu")):
        self.tdnn = ECAPA_TDNN(
            config.input_dim, config.embedding_size, config.mid_channels
        )
        self.voiceprint: dict[str, torch.Tensor] = {}
        self.device = device
        self.config = config

        self.tdnn.to(self.device)

    def embedding(self, x: torch.Tensor):
        self.tdnn.eval()
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.tdnn(x)
        return embedding.squeeze(0)

    def register(self, speaker_id: str, x: torch.Tensor):
        x = x.to(self.device)
        embedding = self.embedding(x)
        self.voiceprint[speaker_id] = embedding.detach().cpu()

    def clean_all(self):
        self.voiceprint.clear()

    def delete_voiceprint(self, speaker_id: str):
        if speaker_id in self.voiceprint:
            del self.voiceprint[speaker_id]
        else:
            print(f"Speaker ID '{speaker_id}' not found in voiceprint database.")

    def find(self, x: torch.Tensor) -> tuple[str, float]:
        if not self.voiceprint:
            raise ValueError(
                "Voiceprint database is empty. Please register at least one speaker before finding."
            )
        x = x.to(self.device)
        embedding = self.embedding(x)
        speaker_ids, voiceprints = zip(*self.voiceprint.items())
        voiceprints = torch.stack(voiceprints).to(self.device)  # [N, D]
        embedding = F.normalize(embedding, dim=0)  # [D]
        voiceprints = F.normalize(voiceprints, dim=1)  # [N, D]
        similarity = embedding @ voiceprints.T  # [N]
        index = int(similarity.argmax().item())
        max_similarity = similarity.max().item()
        return speaker_ids[index], max_similarity

    def save_model(self):
        torch.save(self.tdnn.state_dict(), self.config.tdnn_path)

    def load_model(self):
        self.tdnn.load_state_dict(
            torch.load(self.config.tdnn_path, map_location=self.device)
        )


class SimpleSVTrainer:
    """
    简单说话人识别模型训练器
    """

    def __init__(self, model: SimpleSV, config: TrainConfig):
        self.model = model
        self.config = config

        self.classifier = AAMClassifier(
            self.model.config.embedding_size,
            config.num_classes,
            config.margin,
            config.scale,
        ).to(self.model.device)

        self.tdnn_optimizer = torch.optim.Adam(
            self.model.tdnn.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.tdnn_weight_decay,
        )
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.classifier_weight_decay,
        )

        self.tdnn_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.tdnn_optimizer,
            base_lr=self.config.min_lr,
            max_lr=self.config.max_lr,
            step_size_up=self.config.step_size_up,
            mode="triangular2",
        )
        self.classifier_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.classifier_optimizer,
            base_lr=self.config.min_lr,
            max_lr=self.config.max_lr,
            step_size_up=self.config.step_size_up,
            mode="triangular2",
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def calc_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        logits = self.classifier(embeddings, labels)
        loss = self.criterion(logits, labels.to(self.model.device))
        return loss

    def evaluate(self, dataloader: DataLoader):
        self.model.tdnn.eval()
        self.classifier.eval()
        total_loss = 0.0
        total_counts = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.model.device)
                y = y.to(self.model.device)
                embeddings = self.model.tdnn(x)
                loss = self.calc_loss(embeddings, y)
                total_loss += loss.item() * y.shape[0]
                total_counts += y.shape[0]
        return total_loss / total_counts

    def test(self, enroll_dataset: Dataset, test_dataset: Dataset):
        self.model.tdnn.eval()
        self.model.clean_all()
        for x, y in enroll_dataset:
            x = x.to(self.model.device)
            self.model.register(y, x)

        correct = 0
        total = 0

        for x, y in test_dataset:
            x = x.to(self.model.device)
            pred_speaker_id, _ = self.model.find(x)
            if pred_speaker_id == y:
                correct += 1
            total += 1

        return correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader, first_run: bool):
        step = 0
        epoch = 1
        avg_loss_list = []

        if first_run:
            # 第一次运行会自动清空数据目录，确保训练环境干净
            if os.path.exists(self.config.log_dir):
                shutil.rmtree(self.config.log_dir)
            if os.path.exists(self.config.ckpt_dir):
                shutil.rmtree(self.config.ckpt_dir)

            os.makedirs(self.config.log_dir, exist_ok=True)
            os.makedirs(self.config.ckpt_dir, exist_ok=True)

            with SummaryWriter(self.config.log_dir) as writer:
                writer.add_graph(
                    self.model.tdnn,
                    torch.randn(1, self.model.config.input_dim, 200).to(
                        self.model.device
                    ),
                )

        else:
            # 不是第一次运行则尝试加载最新的checkpoint继续训练
            checkpoint_files = [
                f for f in os.listdir(self.config.ckpt_dir) if f.endswith(".pth")
            ]
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files,
                    key=lambda x: os.path.getctime(
                        os.path.join(self.config.ckpt_dir, x)
                    ),
                )
                checkpoint_path = os.path.join(self.config.ckpt_dir, latest_checkpoint)
                epoch, step, avg_loss_list = self.load_checkpoint(checkpoint_path)
                print(
                    f"Loaded checkpoint '{latest_checkpoint}' (Epoch {epoch}, Step {step})"
                )
            else:
                print("No checkpoint found. Starting training from scratch.")

        while True:
            with SummaryWriter(self.config.log_dir) as writer:
                for x, y in train_loader:
                    self.model.tdnn.train()
                    self.classifier.train()

                    x = x.to(self.model.device)
                    y = y.to(self.model.device)

                    self.tdnn_optimizer.zero_grad()
                    self.classifier_optimizer.zero_grad()
                    embeddings = self.model.tdnn(x)

                    loss = self.calc_loss(embeddings, y)
                    loss.backward()

                    self.tdnn_optimizer.step()
                    self.classifier_optimizer.step()
                    self.tdnn_scheduler.step()
                    self.classifier_scheduler.step()

                    step += 1
                    avg_loss_list.append(loss.item())

                    writer.add_scalar(
                        "Learning Rate",
                        self.tdnn_optimizer.param_groups[0]["lr"],
                        global_step=step,
                    )

                    if step % self.config.eval_freq == 0:
                        avg_loss = sum(avg_loss_list) / len(avg_loss_list)
                        avg_loss_list.clear()
                        val_loss = self.evaluate(val_loader)
                        print(
                            f"Epoch {epoch}, Step {step}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
                        )
                        writer.add_scalars(
                            "Loss",
                            {"Train": avg_loss, "Val": val_loss},
                            global_step=step,
                        )

                    if step % self.config.ckpt_freq == 0:
                        self.save_checkpoint(epoch, step, avg_loss_list)
                        print(f"Checkpoint saved at Epoch {epoch}, Step {step}")

                epoch += 1
                if epoch > self.config.num_epochs:
                    break

        print("Training completed. Saving final model...")
        self.model.save_model()
        print(f"Successfully saved final model: {self.model.config.tdnn_path}")

    def save_checkpoint(self, epoch: int, step: int, avg_loss_list: list[float]):
        checkpoint = {
            "tdnn_state_dict": self.model.tdnn.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "tdnn_optimizer_state_dict": self.tdnn_optimizer.state_dict(),
            "classifier_optimizer_state_dict": self.classifier_optimizer.state_dict(),
            "tdnn_scheduler_state_dict": self.tdnn_scheduler.state_dict(),
            "classifier_scheduler_state_dict": self.classifier_scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "avg_loss_list": avg_loss_list,
        }
        checkpoint_path = os.path.join(
            self.config.ckpt_dir, f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        self.model.tdnn.load_state_dict(checkpoint["tdnn_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.tdnn_optimizer.load_state_dict(checkpoint["tdnn_optimizer_state_dict"])
        self.classifier_optimizer.load_state_dict(
            checkpoint["classifier_optimizer_state_dict"]
        )
        self.tdnn_scheduler.load_state_dict(checkpoint["tdnn_scheduler_state_dict"])
        self.classifier_scheduler.load_state_dict(
            checkpoint["classifier_scheduler_state_dict"]
        )
        return checkpoint["epoch"], checkpoint["step"], checkpoint["avg_loss_list"]
