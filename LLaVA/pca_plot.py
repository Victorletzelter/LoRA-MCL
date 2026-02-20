#%%

import pickle
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root")

def main(args):
    # 1) Resolve config from args
    default_path = f"{os.environ['PROJECT_ROOT']}/pkl_files/audiocaps_relaxed_0_05_beam_size_5.pkl"
    file_path = args.file_path if getattr(args, "file_path", None) else default_path
    translation_plot = bool(getattr(args, "translation_plot", False))

    # 2) Load data
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # 3) Extract candidate strings
    if translation_plot:
        # Expected structure: data['hypothesis_0']['cands'], data['hypothesis_1']['cands']
        cands = [
            data["hypothesis_0"]["cands"],
            data["hypothesis_1"]["cands"],
        ]
        hyp_labels = ["Hypothesis 1", "Hypothesis 2"]
    else:
        # Expected structure: data[0]['hypothesis_k'][i]['cands'][0]
        data0 = data[0]
        cands = []
        hyp_labels = []
        for k in range(5):
            hk = f"hypothesis_{k+1}"
            cands_k = [data0[hk][i]["cands"][0] for i in range(len(data0[hk]))]
            cands.append(cands_k)
            hyp_labels.append(f"Hypothesis {k}")

    # 4) Encode with SentenceTransformer
    model = SentenceTransformer("StyleDistance/styledistance")
    embs = [model.encode(c, convert_to_tensor=False) for c in cands]

    # 5) PCA to 2D
    all_embeddings = torch.cat([torch.tensor(e) for e in embs], dim=0)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings.numpy() if hasattr(all_embeddings, "numpy") else all_embeddings)

    # Split reduced back into groups
    sizes = [len(e) for e in embs]
    pts = []
    start = 0
    for n in sizes:
        pts.append(reduced[start:start + n])
        start += n

    var_explained = pca.explained_variance_ratio_ * 100
    print(f"Variance explained by PC1: {var_explained[0]:.2f}%")
    print(f"Variance explained by PC2: {var_explained[1]:.2f}%")

    # 6) Plot
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman"]

    plt.figure(figsize=(10, 8))
    plt.grid(True, linestyle="--", alpha=0.4)

    markers = ["o", "s", "^", "D", "v"]  # enough for up to 5 hypotheses

    for i, (p, lab) in enumerate(zip(pts, hyp_labels)):
        plt.scatter(
            p[:, 0], p[:, 1],
            label=lab,
            alpha=0.85,
            marker=markers[i % len(markers)],
            edgecolor="black"
        )

    plt.xlabel("PC1", fontsize=30)
    plt.ylabel("PC2", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("pca_hypothesis_candidate_embeddings.png", format="png", dpi=400)
    plt.show()

    # 7) Linear SVM classification
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import numpy as np

    X = np.vstack(embs)
    y = np.concatenate([np.full(len(embs[i]), i) for i in range(len(embs))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Linear SVM Accuracy:", round(acc, 3))
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PCA plot of hypothesis candidate embeddings")

    # Boolean flag: default False, becomes True if provided on CLI
    parser.add_argument(
        "--translation-plot",
        action="store_true",
        help="Use translation pickle structure (data['hypothesis_0/1']['cands']) instead of audio structure."
    )

    # Optional: allow overriding the pickle path from CLI
    parser.add_argument(
        "--file-path",
        type=str,
        default=os.path.join(os.environ["PROJECT_ROOT"], "LLaVA/pkl_files/lora-mcl-2-hyp-translation.pkl"),
        help="Path to the pickle file. If not set, uses the hardcoded default in the script."
    )

    args = parser.parse_args()

    main(args)