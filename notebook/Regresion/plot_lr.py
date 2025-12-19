import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import load_and_prepare_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.colors import ListedColormap


def plot_decision_regions_2d(X, y, classifier, label_encoder):
    markers = ("s", "^", "o", "x", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.02),
        np.arange(x2_min, x2_max, 0.02)
    )

    Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.25, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            c=colors[idx],
            marker=markers[idx],
            label=label_encoder.inverse_transform([cl])[0],
            edgecolor="black"
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_and_prepare_data()

    pca = PCA(n_components=2, random_state=1)
    X_train_pca = pca.fit_transform(data["X_train"])
    X_test_pca = pca.transform(data["X_test"])

    X_all = np.vstack((X_train_pca, X_test_pca))
    y_all = np.hstack((data["y_train"], data["y_test"]))

    model_pca = OneVsRestClassifier(
        LogisticRegression(C=100.0, random_state=1, solver="lbfgs", max_iter=200)
    )
    model_pca.fit(X_train_pca, data["y_train"])

    plot_decision_regions_2d(
        X_all,
        y_all,
        model_pca,
        data["label_encoder"]
    )
