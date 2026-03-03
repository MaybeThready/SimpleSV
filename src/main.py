import torch
import pandas as pd
import torchsummary
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from network.model import SimpleSV, SimpleSVTrainer
from network.config import TDNNConfig, TrainConfig, DatasetConfig
from network.dataset import build_datasets
from rich import print
from rich.rule import Rule
from tensorboard import program


TDNN_NETWORK_PATH = r"D:\Code\AI\SimpleSV\data\model\tdnn.pth"
LOG_DIR = r"D:\Code\AI\SimpleSV\data\model\logs"
CKPT_DIR = r"D:\Code\AI\SimpleSV\data\model\ckpt"
DATASET_ROOT_DIR = r"D:\Code\AI\SimpleSV\data\CN-Celeb\CN-Celeb_flac"
DET_CURVE_DATA_PATH = r"D:\Code\AI\SimpleSV\data\model\det_curve_data.csv"


TDNN_CONFIG = TDNNConfig(tdnn_path=TDNN_NETWORK_PATH)
TRAIN_CONFIG = TrainConfig(
    num_classes=-1,
    log_dir=LOG_DIR,
    ckpt_dir=CKPT_DIR,
)
DATASET_CONFIG = DatasetConfig(
    root_dir=DATASET_ROOT_DIR
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing...")
learn_dataset, enroll_dataset, test_dataset = build_datasets(
    DATASET_CONFIG, split=False
)
TRAIN_CONFIG.num_classes = learn_dataset.data.groupby("speaker_id").size().shape[0]


def main():
    instruct_list = [
        "Show dataset info",
        "Show model info",
        "Train model (first run)",
        "Train model (from checkpoint)",
        "Test model",
        "Show DET curve",
        "Run TensorBoard",
        "Exit"
    ]
    while True:
        print("Available instructions:")
        for i, instruction in enumerate(instruct_list):
            print(f"{i+1}: {instruction}")
        choice = input("> ")

        if choice == "1":
            show_dataset_info()
        elif choice == "2":
            show_model_info()
        elif choice == "3":
            print("This will clean all checkpoints and logs. Are you sure? (y/n)")
            confirm = input("> ")
            if confirm.lower() == "y":
                train(first_run=True)
            else:
                print("Cancelled.")

        elif choice == "4":
            train(first_run=False)
        elif choice == "5":
            test()
        elif choice == "6":
            show_det_curve()
        elif choice == "7":
            run_tensorboard()
        elif choice == "8":
            break
        else:
            print("Invalid choice. Please try again.")


def show_dataset_info():
    pd.set_option("max_colwidth", 100)
    print("Loading dataset...")
    learn_dataset, enroll_dataset, test_dataset = build_datasets(
        DATASET_CONFIG, split=False
    )
    print(Rule("Dataset Info", characters="="))
    print(Rule("Learn Dataset", characters="-"))
    print(f"Learn dataset size: {len(learn_dataset)}")
    print(f"- Train dataset size: {int(len(learn_dataset) * DATASET_CONFIG.train_split_ratio)}")
    print(f"- Validation dataset size: {len(learn_dataset) - int(len(learn_dataset) * DATASET_CONFIG.train_split_ratio)}")
    print(f"Random 5 data in learn dataset:")
    print(learn_dataset.data.iloc[torch.randint(0, len(learn_dataset), (5,))])
    print(f"Average number of samples per speaker: {learn_dataset.data.groupby('speaker_id').size().mean():.2f}")

    print(Rule("Enroll Dataset", characters="-"))
    print(f"Enroll dataset size: {len(enroll_dataset)}")
    print(f"Random 5 data in enroll dataset:")
    print(enroll_dataset.data.iloc[torch.randint(0, len(enroll_dataset), (5,))])

    print(Rule("Test Dataset", characters="-"))
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Random 5 data in test dataset:")
    print(test_dataset.data.iloc[torch.randint(0, len(test_dataset), (5,))])

    print("")


def show_model_info():
    print("Building model...")
    print(Rule("Model Info", characters="="))
    model = SimpleSV(TDNN_CONFIG, DEVICE)
    torchsummary.summary(model.tdnn, (TDNN_CONFIG.input_dim, 200))

    print("")


def train(first_run: bool):
    print("Loading dataset...")
    train_dataset, val_dataset, _, _ = build_datasets(DATASET_CONFIG)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=False)

    print("Building model...")
    model = SimpleSV(TDNN_CONFIG, DEVICE)
    trainer = SimpleSVTrainer(model, TRAIN_CONFIG)

    print(Rule("Training", characters="="))
    trainer.train(train_loader, val_loader, first_run=first_run)

    print("")


def test():
    print("Loading dataset...")
    _, _, enroll_dataset, test_dataset = build_datasets(DATASET_CONFIG)
    print("Building model...")
    model = SimpleSV(TDNN_CONFIG, DEVICE)
    model.load_model()
    trainer = SimpleSVTrainer(model, TRAIN_CONFIG)
    print(Rule("Testing", characters="="))

    trial_list_path = os.path.join(DATASET_CONFIG.root_dir, r"eval\lists\trials.lst")
    with open(trial_list_path, "r") as file:
        trial_list_raw = [line.strip().split() for line in file.readlines()]

    trail_list = []
    for trial in trial_list_raw:
        enroll_file_name, test_file_name, label = trial
        trail_list.append((enroll_file_name + ".flac", test_file_name[5:].replace(".wav", ".flac"), int(label)))

    fa_list, fr_list, threshold_list, eer_threshold, eer = trainer.test(enroll_dataset, test_dataset, trail_list)

    # 保存DET曲线数据
    det_curve_data = pd.DataFrame({
        "threshold": threshold_list,
        "fa": fa_list,
        "fr": fr_list
    })
    det_curve_data.to_csv(DET_CURVE_DATA_PATH, index=False)
    print(f"Test EER: {eer:.4f} at threshold {eer_threshold:.4f}")
    print("DET curve data saved to:", DET_CURVE_DATA_PATH)


def show_det_curve():
    det_curve = pd.read_csv(DET_CURVE_DATA_PATH)
    fa = det_curve["fa"]
    fr = det_curve["fr"]
    threshold = det_curve["threshold"]

    plt.figure(figsize=(8, 6))

    plt.scatter(fa, fr, c=threshold, cmap="winter", s=10)
    plt.plot([0, 1], [0, 1], "k--", label="EER Line")
    plt.colorbar(label="Threshold")

    # 计算EER点
    eer_index = (fa - fr).abs().idxmin()
    eer_threshold = threshold[eer_index]
    eer = (fa[eer_index] + fr[eer_index]) / 2
    plt.scatter(fa[eer_index], fr[eer_index], c="red", s=50, label=f"Threshold {eer_threshold:.4f}")

    # 绘制EER点对应的垂直和水平线
    plt.axvline(x=fa[eer_index], color="red", linestyle="--")
    plt.axhline(y=fr[eer_index], color="red", linestyle="--")

    # 绘制EER点对应的线与坐标轴交点的值标签
    plt.xticks([fa[eer_index], 1], [f"{fa[eer_index]:.4f}", "1.0"])
    plt.yticks([fr[eer_index], 1], [f"{fr[eer_index]:.4f}", "1.0"])

    plt.xlabel("FA Rate")
    plt.ylabel("FR Rate")
    plt.title("DET Curve")
    plt.legend()
    plt.show()


def run_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", TRAIN_CONFIG.log_dir])
    url = tb.launch()
    print(f"TensorBoard is running at {url}")


if __name__ == "__main__":
    main()
