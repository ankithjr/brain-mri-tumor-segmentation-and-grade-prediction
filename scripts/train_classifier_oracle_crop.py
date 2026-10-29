from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
ROOT = Path("/scratch/markov3/users/arj94/brain-glioma")
INPUT = ROOT / "preprocessed/classifier_oracle_crop_v1"
OUTPUT = ROOT / "checkpoints/grade_classifier_oracle_crop"
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 15
BATCH_SIZE = 8
class MRIDataset(Dataset):
    def __init__(self, split):
        self.frame = pd.read_csv(ROOT / f"results/splits/{split}.csv")
        self.folder = INPUT / split
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.folder / f"{row['exam_id']}.npz"
        with np.load(path) as data:
            image = data["image"].astype(np.float32)
            grade = int(data["grade"])
        if grade != int(row["grade"]):
            raise ValueError(f"Grade mismatch: {path}")
        if image.shape != (4, 96, 96, 64) or not np.isfinite(image).all():
            raise ValueError(f"Invalid image: {path}")
        return torch.from_numpy(image), grade - 2
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(4, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(128, 3)
    def forward(self, x):
        return self.head(self.features(x))
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    train_data = MRIDataset("train")
    val_data = MRIDataset("validation")
    assert len(train_data) == 346 and len(val_data) == 74
    assert not set(train_data.frame.patient_id) & set(val_data.frame.patient_id)
    counts = np.array([
        (train_data.frame["grade"] == grade).sum()
        for grade in (2, 3, 4)
    ])
    assert np.all(counts > 0)
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; stopping: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    device = torch.device("cuda")
    sample_weights = [1.0 / counts[int(grade) - 2] for grade in train_data.frame["grade"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    history = []
    best_f1 = -1.0
    stale = 0
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Training counts, grades 2/3/4:", counts.tolist(), flush=True)
    print("Sampler: inverse class frequency; loss: unweighted cross-entropy", flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        model.eval()
        truth, predictions = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images.to(device, non_blocking=True))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                truth.extend(targets.tolist())
        macro_f1 = float(f1_score(
            truth, predictions, labels=[0, 1, 2],
            average="macro", zero_division=0
        ))
        balanced_acc = float(balanced_accuracy_score(truth, predictions))
        matrix = confusion_matrix(
            truth, predictions, labels=[0, 1, 2]
        ).tolist()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_data),
            "validation_macro_f1": macro_f1,
            "validation_balanced_accuracy": balanced_acc,
            "validation_confusion_matrix": matrix,
        }
        history.append(record)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            stale = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_macro_f1": macro_f1,
                "seed": SEED,
                "grade_mapping": {"0": 2, "1": 3, "2": 4},
                "input_shape": [4, 96, 96, 64],
            }, OUTPUT / "checkpoint_best.pt")
        else:
            stale += 1
        (OUTPUT / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"train loss {record['train_loss']:.4f} | "
            f"val macro-F1 {macro_f1:.4f} | "
            f"val balanced accuracy {balanced_acc:.4f} | "
            f"best F1 {best_f1:.4f} | confusion matrix {matrix}",
            flush=True,
        )
        if stale >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break
    print(f"TRAINING COMPLETE; best validation macro-F1: {best_f1:.4f}")
    print(f"Results: {OUTPUT}")
if __name__ == "__main__":
    main()
