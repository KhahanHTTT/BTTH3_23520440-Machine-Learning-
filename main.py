import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from svm import SoftMarginSVM

DATA_DIR = Path("chest_xray")
IMG_SIZE = (128, 128)
LABEL_MAP = {"NORMAL": -1, "PNEUMONIA": 1}

C = 0.01
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 64
MAX_PER_CLASS = 1000    # test nhanh

def load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for class_name, label in LABEL_MAP.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Directory {class_dir} not found")
        
        files = (
            sorted(class_dir.glob("*.jpeg"))
            + sorted(class_dir.glob("*.jpg"))
            + sorted(class_dir.glob("*.png"))
        )
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]

        for fpath in files:
            img = Image.open(fpath).convert("L").resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
            X_list.append(arr)
            y_list.append(label)
        
    return np.stack(X_list), np.array(y_list, dtype=np.float32)

def load_dataset():
    print("Loading dataset...")
    t0 = time.time()
    X_train, y_train = load_split(DATA_DIR / "train")
    X_val, y_val = load_split(DATA_DIR / "val")
    X_test, y_test = load_split(DATA_DIR / "test")
    print(f"    Dataset loaded in {time.time() - t0:.1f} seconds")
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Val: {X_val.shape[0]} samples")
    print(f"    Test: {X_test.shape[0]} samples")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    if name:
        print(f"    {name}")
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Precision: {prec:.4f}")
    print(f"{name} Recall: {rec:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

#Assignment 1 - NumPy SVM

def run_assignment_1(X_train, y_train, X_test, y_test):
    print("\nRunning Assignment 1 - NumPy SVM")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    
    model = SoftMarginSVM(
        C=float(C),
        lr=float(LR),
        n_epochs=int(N_EPOCHS),
        batch_size=int(BATCH_SIZE),
        random_state=42,
        verbose=True,
        print_every=1,
    )

    print(f"\nTraining SoftMarginSVM  (C={C}, lr={LR}, epochs={N_EPOCHS}, batch_size={BATCH_SIZE})...\n")
    t0 = time.time()
    model.fit(X_tr_s, y_train)
    print(f"\n  Training completed in {time.time() - t0:.1f} seconds")

    y_pred = model.predict(X_te_s)
    metrics = evaluate(y_true=y_test, y_pred=y_pred, name="NumPy SVM")
    return metrics, model, scaler


#Assignment 2 - scikit-learn SVM

def run_assignment_2(X_train, y_train, X_test, y_test, scaler):
    print("\nRunning Assignment 2 - scikit-learn SVM")

    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    print(f"\nTraining scikit-learn SVM  (C={C}, kernel=linear)...\n")
    t0 = time.time()
    clf = SVC(C=C, kernel="linear", random_state=42, max_iter=2000)
    clf.fit(X_tr_s, y_train)
    print(f"\n  Training completed in {time.time() - t0:.1f} seconds")

    y_pred = clf.predict(X_te_s)
    metrics = evaluate(y_true=y_test, y_pred=y_pred, name="scikit-learn SVM")
    return metrics

def compare(m1: dict, m2: dict):
    print("\nComparison of Models:")
    print(f"{'Metric':<10} {'NumPy SVM':>18} {'scikit-learn SVM':>12}")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key:<12} {m1[key]*100:>11.2f}%  {m2[key]*100:>11.2f}%")


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()
    metrics_1, numpy_model, scaler = run_assignment_1(X_train, y_train, X_test, y_test)
    metrics_2 = run_assignment_2(X_train, y_train, X_test, y_test, scaler)
    compare(metrics_1, metrics_2)