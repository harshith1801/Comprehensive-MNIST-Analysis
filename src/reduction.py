import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import umap.umap_ as umap

def apply_pca(X_train, X_test, n_components=40):
    """Applies PCA to the training and test sets."""
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced, X_test_reduced

def apply_kpca(X_train, X_test, n_components=40):
    """Applies Kernel PCA to the training and test sets."""
    print(f"Applying Kernel PCA with {n_components} components...")
    # Use a subsample for fitting due to computational cost
    subsample_idx = np.random.choice(len(X_train), 5000, replace=False)

    kpca = KernelPCA(n_components=n_components, kernel='rbf', random_state=42)
    kpca.fit(X_train[subsample_idx])

    X_train_reduced = kpca.transform(X_train)
    X_test_reduced = kpca.transform(X_test)
    return X_train_reduced, X_test_reduced

def apply_umap(X_train, X_test, n_components=40):
    """Applies UMAP to the training and test sets."""
    print(f"Applying UMAP with {n_components} components...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=15, random_state=42)
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    return X_train_reduced, X_test_reduced