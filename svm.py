import numpy as np

class SoftMarginSVM:

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 64,
        random_state: int = 42,
        verbose: bool = True,
        print_every: int = 1,
    ):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.print_every = print_every

        self.w: np.ndarray | None = None
        self.b: float = 0.0

        self.history: dict[str, list[float]] = {
            "loss": [],
            "train_acc": [],
        }

    def _hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        margins = y * (X @ self.w + self.b)
        hinge = np.maximum(0.0, 1.0 - margins)
        loss = 0.5 * float(self.w @ self.w) + self.C * hinge.mean()
        return loss
    
    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftMarginSVM":
        rgn = np.random.default_rng(self.random_state)
        N, d = X.shape

        self.w = rgn.normal(0.0, 0.01, size = d)
        self.b = 0.0

        for epoch in range(1, self.n_epochs + 1):
            idx = rgn.permutation(N)
            X_shuf, y_shuf = X[idx], y[idx]

            for start in range(0, N, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                margins = yb * (Xb @ self.w + self.b)
                violated = margins < 1

                dw = self.w.copy()
                db = 0.0
                if violated.any():
                    dw -= self.C * (yb[violated, None] * Xb[violated]).mean(axis = 0)
                    db -= self.C * yb[violated].mean()
                
                self.w -= self.lr * dw
                self.b -= self.lr * db

            loss = self._hinge_loss(X, y)
            acc = self._accuracy(X, y)
            self.history["loss"].append(loss)
            self.history["train_acc"].append(acc)

            if self.verbose and (epoch %  self.print_every == 0 or epoch == 1):
                print(
                    f"  [NumPy SVM]Epoch {epoch:>3d}/{self.n_epochs}"
                    f"  |   Loss: {loss:.4f}"
                    f"  |   Train Acc: {acc*100:.2f}%"
                )

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.w + self.b
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


        