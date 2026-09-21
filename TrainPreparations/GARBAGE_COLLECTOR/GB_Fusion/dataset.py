import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================================================
# LOAD SINGLE NPZ
# =========================================================

def load_recording(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    X_context = data["X_context"]
    X_embed = data["X_embed"]

    y = data["y"]

    feature_keys = list(
        data["feature_keys"]
    )

    return (
        X_context,
        X_embed,
        y,
        feature_keys
    )


# =========================================================
# LOAD SPLIT
# =========================================================

def load_split(folder):

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".npz")
    ])

    X_context_all = []
    X_embed_all = []
    y_all = []

    feature_keys = None

    for f in files:

        (
            X_context,
            X_embed,
            y,
            keys
        ) = load_recording(f)

        X_context_all.append(X_context)
        X_embed_all.append(X_embed)
        y_all.append(y)

        if feature_keys is None:
            feature_keys = keys

    X_context_all = np.concatenate(
        X_context_all,
        axis=0
    )

    X_embed_all = np.concatenate(
        X_embed_all,
        axis=0
    )

    y_all = np.concatenate(
        y_all,
        axis=0
    )

    return (
        X_context_all,
        X_embed_all,
        y_all,
        feature_keys
    )


# =========================================================
# PCA + FUSION
# =========================================================

class FusionPreprocessor:

    def __init__(
        self,
        pca_dim=64
    ):

        self.pca_dim = pca_dim

        self.context_imputer = (
            SimpleImputer(
                strategy="constant",
                fill_value=0.0
            )
        )

        self.context_scaler = (
            StandardScaler()
        )

        self.embed_scaler = (
            StandardScaler()
        )

        self.pca = PCA(
            n_components=pca_dim,
            random_state=42
        )

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        X_context,
        X_embed
    ):

        # -----------------------------
        # context
        # -----------------------------

        X_context = (
            self.context_imputer
            .fit_transform(X_context)
        )

        X_context = (
            self.context_scaler
            .fit_transform(X_context)
        )

        # -----------------------------
        # embeddings
        # -----------------------------

        X_embed = (
            self.embed_scaler
            .fit_transform(X_embed)
        )

        self.pca.fit(X_embed)

        return self

    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(
        self,
        X_context,
        X_embed
    ):

        # -----------------------------
        # context
        # -----------------------------

        X_context = (
            self.context_imputer
            .transform(X_context)
        )

        X_context = (
            self.context_scaler
            .transform(X_context)
        )

        # -----------------------------
        # embeddings
        # -----------------------------

        X_embed = (
            self.embed_scaler
            .transform(X_embed)
        )

        X_embed = (
            self.pca
            .transform(X_embed)
        )

        # -----------------------------
        # fusion
        # -----------------------------

        X_fused = np.concatenate(
            [
                X_context,
                X_embed
            ],
            axis=1
        )

        return X_fused

    # =====================================================
    # FIT TRANSFORM
    # =====================================================

    def fit_transform(
        self,
        X_context,
        X_embed
    ):

        self.fit(
            X_context,
            X_embed
        )

        return self.transform(
            X_context,
            X_embed
        )