import io
import time
import numpy as np
import streamlit as st
from PIL import Image

MAX_DIM = 300


class KMeansNumPy:
    """
    K-Means++ with multiple restarts. All distance calculations are
    vectorised — no Python loops over individual pixels.
    """

    def __init__(self, k=5, max_iters=100, tol=1e-4, n_init=3, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = float("inf")

    def _init_centroids(self, X, rng):
        """
        K-Means++ init: each new centroid is sampled with probability
        proportional to its squared distance from the nearest existing centroid.
        This spreads centroids apart and reduces iterations vs. random init.
        """
        n_samples = X.shape[0]
        centroids = [X[rng.integers(0, n_samples)]]

        for _ in range(1, self.k):
            c = np.array(centroids)
            # (n,1,d) - (1,c,d) -> (n,c,d) -> (n,c) squared distances
            sq_dists = ((X[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2).sum(axis=2)
            min_sq_dists = sq_dists.min(axis=1)
            probs = min_sq_dists / min_sq_dists.sum()
            centroids.append(X[rng.choice(n_samples, p=probs)])

        return np.array(centroids)

    def _assign_labels(self, X, centroids):
        """
        Assign each point to its nearest centroid.
        Broadcasting: (n,1,d) - (1,k,d) -> (n,k,d) -> (n,k) -> argmin -> (n,)
        """
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return ((diff ** 2).sum(axis=2)).argmin(axis=1)

    def _update_centroids(self, X, labels, old_centroids):
        new_centroids = np.empty_like(old_centroids)
        for c in range(self.k):
            mask = labels == c
            # Keep previous centroid if a cluster is empty to avoid NaN
            new_centroids[c] = X[mask].mean(axis=0) if mask.any() else old_centroids[c]
        return new_centroids

    def fit_predict(self, X):
        rng = np.random.default_rng(self.random_state)
        best_labels, best_centroids, best_inertia = None, None, float("inf")

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, rng)

            for _ in range(self.max_iters):
                labels = self._assign_labels(X, centroids)
                new_centroids = self._update_centroids(X, labels, centroids)
                if np.linalg.norm(new_centroids - centroids, axis=1).max() < self.tol:
                    centroids = new_centroids
                    break
                centroids = new_centroids

            inertia = float(((X - centroids[labels]) ** 2).sum())
            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels, centroids

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        return best_labels


@st.cache_data(show_spinner=False)
def load_and_resize(file_bytes):
    """Cache is keyed on file_bytes, so the image isn't reloaded on every slider move."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    scale = min(MAX_DIM / max(w, h), 1.0)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


@st.cache_data(show_spinner=False)
def extract_color_features(img_array):
    """Flatten to (H*W, 3) and normalise RGB to [0, 1]."""
    return img_array.reshape(-1, 3).astype(np.float64) / 255.0


@st.cache_data(show_spinner=False)
def extract_spatial_features(img_array):
    """
    Build (H*W, 5) feature matrix: [r, g, b, x_norm, y_norm].

    Spatial coordinates are cached without lambda applied so the same
    cached array is reused across different lambda values. The caller
    scales cols 3-4 by lambda after copying.
    """
    H, W, _ = img_array.shape
    rgb = img_array.reshape(-1, 3).astype(np.float64) / 255.0

    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_norm = (cols.ravel() / max(W - 1, 1)).reshape(-1, 1)
    y_norm = (rows.ravel() / max(H - 1, 1)).reshape(-1, 1)

    return np.concatenate([rgb, x_norm, y_norm], axis=1)


def reconstruct_image(labels, centroids, H, W):
    """
    Map each pixel's cluster label to its centroid's RGB colour.

    centroids[labels] gathers the centroid row for every pixel in one
    vectorised step, replacing the per-pixel colour with the cluster mean.
    Result is clipped and cast to uint8 before reshaping to (H, W, 3).
    """
    rgb_centroids = centroids[:, :3]                       # drop spatial dims if present
    pixel_colors = np.clip(rgb_centroids[labels] * 255.0, 0, 255).astype(np.uint8)
    return pixel_colors.reshape(H, W, 3)


def main():
    st.set_page_config(page_title="K-Means Image Segmentation", layout="wide")

    st.title("Image Segmentation Using K-Means Clustering")
    st.markdown(
        "Segments an uploaded image into K colour regions using a custom NumPy "
        "K-Means implementation. Upload an image in the sidebar to begin."
    )
    
    with st.expander("📖 Guide & Legend: Understanding the Parameters"):
        st.markdown("""
        ### Feature Pipelines
        * **Color Only (RGB):** The algorithm exclusively analyzes pixel colors. Pixels with similar colors are grouped into the same cluster, even if they are physically located on opposite sides of the image.
        * **Color + Spatial (RGB + XY):** The algorithm analyzes both the color and the physical $(x, y)$ location of the pixel. This encourages the K-Means algorithm to group pixels into contiguous, solid spatial blocks rather than scattered fragments.

        ### Important Controls
        * **Number of Clusters (K):** Determines exactly how many distinct regions the algorithm will segment the image into.
        * **Spatial Weight ($\\lambda$):** *Only active in 'Color + Spatial' mode.* This multiplier dictates how heavily the $(x, y)$ coordinates are penalized compared to the RGB colors. 
        * **Low Weight (e.g., 0.1):** Color similarity remains the dominant factor.
        * **High Weight (e.g., 5.0):** Physical proximity overrides color similarity, forcing clusters into tightly bound geographic regions regardless of texture.
        """)
        
    st.divider()

    with st.sidebar:
        st.header("Controls")

        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        st.subheader("K-Means Parameters")

        k = st.slider("Number of Clusters (K)", min_value=2, max_value=15, value=5)

        pipeline = st.radio(
            "Feature Pipeline",
            options=["Color Only", "Color + Spatial"],
            help=(
                "**Color Only**: clusters purely by RGB similarity.\n\n"
                "**Color + Spatial**: includes pixel (x, y) coordinates so "
                "nearby pixels are more likely to share a cluster."
            ),
        )

        lam = 1.0
        if pipeline == "Color + Spatial":
            lam = st.slider(
                "Spatial Weight (λ)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Scales the (x, y) features. Higher λ = position outweighs colour.",
            )

        st.divider()
        st.caption(f"Images resized to {MAX_DIM}px max. K-Means++, 3 restarts, tol=1e-4.")

    if uploaded_file is None:
        st.info("Upload an image in the sidebar to get started.")
        return

    file_bytes = uploaded_file.read()
    img_array = load_and_resize(file_bytes)
    H, W, _ = img_array.shape

    if pipeline == "Color Only":
        features = extract_color_features(img_array)
    else:
        # .copy() because we mutate cols 3-4 in-place with lambda scaling
        features = extract_spatial_features(img_array).copy()
        features[:, 3] *= lam  # x_norm scaled by λ
        features[:, 4] *= lam  # y_norm scaled by λ

    with st.spinner("Running K-Means..."):
        t0 = time.perf_counter()
        km = KMeansNumPy(k=k, max_iters=100, tol=1e-4, n_init=3, random_state=42)
        labels = km.fit_predict(features)
        elapsed = time.perf_counter() - t0

    segmented = reconstruct_image(labels, km.centroids_, H, W)

    col_orig, col_seg = st.columns(2, gap="large")
    with col_orig:
        st.subheader("Original")
        st.image(img_array, use_container_width=True)
        st.caption(f"{W} x {H} px")
    with col_seg:
        st.subheader(f"Segmented  (K = {k})")
        st.image(segmented, use_container_width=True)
        caption = f"Pipeline: {pipeline}"
        if pipeline == "Color + Spatial":
            caption += f"  |  λ = {lam:.1f}"
        st.caption(caption)

    st.success(f"Done in {elapsed:.2f}s  |  inertia = {km.inertia_:.2f}")

    with st.expander("Cluster Colour Palette"):
        rgb_centroids = np.clip(km.centroids_[:, :3] * 255, 0, 255).astype(np.uint8)
        swatch_cols = st.columns(k)
        for i, col in enumerate(swatch_cols):
            r, g, b = rgb_centroids[i]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            col.markdown(
                f"<div style='background:{hex_color};height:50px;border-radius:6px;"
                f"border:1px solid #ccc'></div>",
                unsafe_allow_html=True,
            )
            col.caption(hex_color)


if __name__ == "__main__":
    main()