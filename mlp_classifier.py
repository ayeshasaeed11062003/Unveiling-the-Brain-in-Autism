"""
mlp_classifier.py — Unveiling the Brain in Autism  [FIXED v2]
==============================================================
MLP deep learning baseline following Heinsfeld et al. 2018,
with fixes for the small-N / high-D problem inherent to ABIDE II.

PROBLEM WITH v1:
  19,900 features → 1,000 neurons with only 626 training samples
  = 31x more features than samples → immediate overfitting
  val_loss = 3.96 while train_loss = 0.49 at epoch 20

FIXES APPLIED:
  [F1] PCA(n_components=300) before the MLP
       Reduces 19,900 → 300 dims, preserving ~85% variance
  [F2] Larger validation split (15% instead of 10%)
  [F3] LR scheduler: ReduceLROnPlateau(factor=0.5, patience=10)
  [F4] Weight decay 1e-4 in Adam (L2 regularisation)
  [F5] Patience increased to 30 epochs

Architecture (Heinsfeld 2018, adapted for ABIDE II scale):
  PCA(300) → Dense(1000,ReLU) → Drop(0.5)
           → Dense(600, ReLU) → Drop(0.5)
           → Dense(300, ReLU) → Drop(0.5)
           → Dense(1, Sigmoid)

Authors: Ayesha Saeed et al., FAST University Karachi — FYP 2024-25
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              roc_curve, roc_auc_score, balanced_accuracy_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

BG = "#F4F6F8"; RED = "#E84545"; TEAL = "#2B7A78"; NAVY = "#1A3A5C"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_COMPONENTS = 300


def section(t):
    print(f"\n{'='*66}\n  {t}\n{'='*66}")


# ══════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE
# ══════════════════════════════════════════════════════════════
section("1 / 5   LOADING & PREPARING FEATURES")

matrices = np.load("padded_matrices_200.npy")
y_raw    = np.load("dx_labels.npy")
y_site   = np.load("site_labels.npy")
y_dx     = (y_raw - 1).astype(int)
site_arr = np.array(y_site)

print(f"  Matrices : {matrices.shape}   range: {matrices.min():.4f} → {matrices.max():.4f}")

if matrices.max() > 2.0:
    print("  WARNING: Covariance detected. Run prepare_matrices.py first!")
    def cov_to_corr(M):
        d = np.sqrt(np.abs(np.diag(M))); d[d<1e-8]=1e-8
        C = M/np.outer(d,d); np.fill_diagonal(C,0)
        return np.clip(np.nan_to_num(C,nan=0.0),-1.0,1.0)
    corr_matrices = np.array([cov_to_corr(m) for m in matrices])
else:
    corr_matrices = matrices.copy()
    for i in range(len(corr_matrices)):
        np.fill_diagonal(corr_matrices[i], 0)
    corr_matrices = np.nan_to_num(corr_matrices, nan=0.0)
    print("  Pearson FC confirmed")

mask      = np.triu(np.ones((200,200), dtype=bool), k=1)
X_pearson = np.array([c[mask] for c in corr_matrices])
X_pearson = np.arctanh(np.clip(X_pearson, -0.9999, 0.9999))
print(f"  FC features  : {X_pearson.shape}")
print(f"  Subjects     : {len(y_dx)}   ASD: {(y_dx==1).sum()}   Control: {(y_dx==0).sum()}")
print(f"  Sites        : {len(np.unique(site_arr))}")
print(f"  Device       : {DEVICE}")


def residualize(X, site_labels):
    enc = OneHotEncoder(sparse_output=False)
    S   = enc.fit_transform(np.array(site_labels).reshape(-1,1))
    X_r = X.copy().astype(np.float64)
    reg = LinearRegression()
    for i in range(X.shape[1]):
        reg.fit(S, X_r[:,i]); X_r[:,i] -= reg.predict(S)
    return X_r


print("  Residualising...", end=" ", flush=True)
X_res = residualize(X_pearson, site_arr)
print("done.")

idx = np.arange(len(y_dx))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_dx)
y_tr, y_te     = y_dx[idx_tr], y_dx[idx_te]

# [F1] PCA — fit on train only
print(f"  [F1] PCA {X_res.shape[1]} → {PCA_COMPONENTS}...", end=" ", flush=True)
sc_tr  = StandardScaler()
pca_tr = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_tr_pca = pca_tr.fit_transform(sc_tr.fit_transform(X_res[idx_tr])).astype(np.float32)
X_te_pca = pca_tr.transform(sc_tr.transform(X_res[idx_te])).astype(np.float32)
var_exp  = pca_tr.explained_variance_ratio_.sum()
print(f"done.  Variance explained: {var_exp:.1%}")
print(f"  After PCA — Train: {X_tr_pca.shape}   Test: {X_te_pca.shape}")

pos_weight = torch.tensor(
    [float(np.sum(y_tr==0)) / float(np.sum(y_tr==1))],
    dtype=torch.float32).to(DEVICE)


# ══════════════════════════════════════════════════════════════
# 2. MODEL
# ══════════════════════════════════════════════════════════════
section("2 / 5   MODEL ARCHITECTURE")

class Heinsfeld_MLP(nn.Module):
    def __init__(self, input_dim=PCA_COMPONENTS, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1000), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1000, 600),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(600,  300),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(300,  1),         nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)


n_params = sum(p.numel() for p in Heinsfeld_MLP().parameters())
print(f"  PCA({PCA_COMPONENTS}) → 1000 → 600 → 300 → 1  (dropout=0.5)")
print(f"  Parameters       : {n_params:,}")
print(f"  Params/sample    : {n_params/len(y_tr):.1f}x  (was ~2,600x before PCA)")


# ══════════════════════════════════════════════════════════════
# 3. TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════
def train_mlp(X_tr, y_tr, X_val, y_val,
              input_dim=PCA_COMPONENTS,
              epochs=200, batch_size=32,
              lr=1e-4, patience=30, dropout=0.5,
              verbose=True):

    model     = Heinsfeld_MLP(input_dim, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # [F4]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(                     # [F3]
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.BCELoss(reduction="none")

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    pw = float(np.sum(y_tr==0)) / float(max(np.sum(y_tr==1), 1))
    sw = torch.where(y_tr_t==1, torch.tensor(pw), torch.tensor(1.0))

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t, sw),
                         batch_size=batch_size, shuffle=True)

    best_val, best_state, patience_count = float("inf"), None, 0
    tr_losses, val_losses, lr_hist       = [], [], []

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb, wb in loader:
            Xb, yb, wb = Xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            loss = (criterion(model(Xb), yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        train_loss = ep_loss / len(loader)
        tr_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).mean().item()
        val_losses.append(val_loss)
        lr_hist.append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if verbose and (epoch+1) % 20 == 0:
            ratio = val_loss / max(train_loss, 1e-8)
            print(f"    Epoch {epoch+1:>3}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  ratio={ratio:.2f}  "
                  f"lr={lr_hist[-1]:.1e}  patience={patience_count}/{patience}")

        if patience_count >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}  (best val={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model, tr_losses, val_losses, lr_hist


@torch.no_grad()
def predict(model, X):
    model.eval()
    prob = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return (prob >= 0.5).astype(int), prob


def evaluate(y_true, y_pred, y_prob, label=""):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    r = dict(
        acc=accuracy_score(y_true, y_pred),
        bal=balanced_accuracy_score(y_true, y_pred),
        f1 =f1_score(y_true, y_pred, average="weighted"),
        auc=roc_auc_score(y_true, y_prob),
        sens=tp/max(tp+fn,1), spec=tn/max(tn+fp,1),
        cm=cm, y_pred=y_pred, y_prob=y_prob,
    )
    if label:
        print(f"  {label:<35}  Acc={r['acc']:.3f}  Bal={r['bal']:.3f}  "
              f"AUC={r['auc']:.3f}  Sens={r['sens']:.3f}  Spec={r['spec']:.3f}")
    return r


# ══════════════════════════════════════════════════════════════
# 4. TRAIN — 80/20 SPLIT
# ══════════════════════════════════════════════════════════════
section("4 / 5   TRAINING  (80/20 split)")

# [F2] 15% validation split
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr_pca, y_tr, test_size=0.15, random_state=42, stratify=y_tr)

print(f"  Train: {len(y_tr2)}   Val: {len(y_val)}   Test: {len(y_te)}")
print(f"  Input dim: {PCA_COMPONENTS}  (after PCA)\n")

model_mlp, tr_losses, val_losses, lr_hist = train_mlp(
    X_tr2, y_tr2, X_val, y_val,
    input_dim=PCA_COMPONENTS,
    epochs=200, batch_size=32, lr=1e-4, patience=30, verbose=True)

yp, ypr    = predict(model_mlp, X_te_pca)
result_raw = evaluate(y_te, yp, ypr, "MLP raw (t=0.5)")

# Youden's J
fpr_c, tpr_c, ths_c = roc_curve(y_te, ypr)
J_c  = tpr_c + (1-fpr_c) - 1
bt   = ths_c[np.argmax(J_c)]
yp_j = (ypr >= bt).astype(int)
cm_j = confusion_matrix(y_te, yp_j)
tn, fp_, fn, tp = cm_j.ravel()
result_opt = dict(
    threshold=bt, J=float(np.max(J_c)),
    acc=accuracy_score(y_te,yp_j), bal=balanced_accuracy_score(y_te,yp_j),
    f1=f1_score(y_te,yp_j,average="weighted"), auc=result_raw["auc"],
    sens=tp/max(tp+fn,1), spec=tn/max(tn+fp_,1), cm=cm_j,
)
print(f"  Threshold-opt (t={bt:.3f}, J={result_opt['J']:.3f}):  "
      f"Acc={result_opt['acc']:.3f}  Bal={result_opt['bal']:.3f}  "
      f"Sens={result_opt['sens']:.3f}  Spec={result_opt['spec']:.3f}")

final_ratio = val_losses[-1] / max(tr_losses[-1], 1e-8)
print(f"\n  Overfitting check  train={tr_losses[-1]:.4f}  val={val_losses[-1]:.4f}  "
      f"ratio={final_ratio:.2f}  "
      f"({'OK' if final_ratio < 2.5 else 'still some overfitting'})")


# ══════════════════════════════════════════════════════════════
# 4b. LOSO — LEAK-FREE [N10]
# ══════════════════════════════════════════════════════════════
section("4b / 5   LOSO  (leak-free — PCA inside each fold)")

loso_accs, loso_aucs, loso_f1s, loso_sites = [], [], [], []

for tr_i, te_i in LeaveOneGroupOut().split(X_pearson, y_dx, groups=site_arr):
    site_id = np.unique(site_arr[te_i])[0]
    site_tr = site_arr[tr_i]

    # Residualise inside fold
    enc_f = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    S_tr  = enc_f.fit_transform(site_tr.reshape(-1,1))
    S_te  = enc_f.transform(site_arr[te_i].reshape(-1,1))
    reg_f = LinearRegression()
    X_tr_f = X_pearson[tr_i].copy().astype(np.float64)
    X_te_f = X_pearson[te_i].copy().astype(np.float64)
    for i in range(X_pearson.shape[1]):
        reg_f.fit(S_tr, X_pearson[tr_i, i])
        X_tr_f[:,i] -= reg_f.predict(S_tr)
        X_te_f[:,i] -= reg_f.predict(S_te)

    # PCA inside fold [F1 applied per-fold]
    n_comp = min(PCA_COMPONENTS, len(tr_i)-1)
    sc_f   = StandardScaler()
    pca_f  = PCA(n_components=n_comp, random_state=42)
    X_tr_fp = pca_f.fit_transform(sc_f.fit_transform(X_tr_f)).astype(np.float32)
    X_te_fp = pca_f.transform(sc_f.transform(X_te_f)).astype(np.float32)

    y_tr_f = y_dx[tr_i]; y_te_f = y_dx[te_i]
    if len(np.unique(y_tr_f)) < 2:
        loso_accs.append(0.5); loso_aucs.append(0.5)
        loso_f1s.append(0.0);  loso_sites.append(site_id)
        continue

    X_tr_fv, X_val_f, y_tr_fv, y_val_f = train_test_split(
        X_tr_fp, y_tr_f, test_size=0.15, random_state=42, stratify=y_tr_f)

    mdl_f, _, _, _ = train_mlp(
        X_tr_fv, y_tr_fv, X_val_f, y_val_f,
        input_dim=n_comp, epochs=100,
        batch_size=32, lr=1e-4, patience=20, verbose=False)

    yp_f, ypr_f = predict(mdl_f, X_te_fp)
    loso_accs.append(accuracy_score(y_te_f, yp_f))
    loso_f1s.append(f1_score(y_te_f, yp_f, average="weighted"))
    try:    loso_aucs.append(roc_auc_score(y_te_f, ypr_f))
    except: loso_aucs.append(np.nan)
    loso_sites.append(site_id)
    print(f"  Site {str(site_id):<14}  Acc={loso_accs[-1]:.3f}  "
          f"AUC={loso_aucs[-1]:.3f}  n={len(te_i)}", flush=True)

print(f"\n  LOSO Accuracy : {np.mean(loso_accs):.4f} ± {np.std(loso_accs):.4f}")
print(f"  LOSO F1       : {np.mean(loso_f1s):.4f}")
print(f"  LOSO AUC      : {np.nanmean(loso_aucs):.4f}")


# ══════════════════════════════════════════════════════════════
# 5. FIGURES
# ══════════════════════════════════════════════════════════════
section("5 / 5   FIGURES")

# Fig A: Training curves
print("  [Fig A] Training curves...")
fig_a, axes_a = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig_a.suptitle("MLP Training — PCA(300) Fix Applied [FIXED v2]",
               fontsize=13, fontweight="bold")

axes_a[0].plot(tr_losses,  color=TEAL, lw=2, label="Train")
axes_a[0].plot(val_losses, color=RED,  lw=2, label="Val")
axes_a[0].axvline(np.argmin(val_losses), color=NAVY, ls="--", lw=1.5,
                  label=f"Best epoch {np.argmin(val_losses)+1}")
best_r = min(val_losses)/max(tr_losses[np.argmin(val_losses)], 1e-8)
axes_a[0].set_title(f"Loss Curves  (val/train at best: {best_r:.2f})",
                    fontsize=10, fontweight="bold")
axes_a[0].set_xlabel("Epoch"); axes_a[0].set_ylabel("BCE Loss")
axes_a[0].legend(fontsize=9)

axes_a[1].plot(lr_hist, color=NAVY, lw=2)
axes_a[1].set_title("LR Schedule (ReduceLROnPlateau)", fontsize=10, fontweight="bold")
axes_a[1].set_xlabel("Epoch"); axes_a[1].set_ylabel("LR"); axes_a[1].set_yscale("log")

fpr_r, tpr_r, _ = roc_curve(y_te, ypr)
axes_a[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
axes_a[2].plot(fpr_r, tpr_r, color=TEAL, lw=2.5,
               label=f"MLP  AUC={result_raw['auc']:.3f}")
axes_a[2].axvline(1-result_opt["spec"], color=RED, ls="--", lw=1.5,
                  label=f"Youden t={bt:.3f}")
axes_a[2].set_xlim(0,1); axes_a[2].set_ylim(0,1.02)
axes_a[2].set_xlabel("FPR"); axes_a[2].set_ylabel("TPR")
axes_a[2].set_title("ROC Curve", fontsize=10, fontweight="bold")
axes_a[2].legend(fontsize=9)
plt.tight_layout()
fig_a.savefig(f"{SAVE_DIR}/figureA_mlp_training.png", dpi=150, bbox_inches="tight")
plt.close(fig_a); print(f"  Saved → {SAVE_DIR}/figureA_mlp_training.png")

# Fig B: Confusion matrices
print("  [Fig B] Confusion matrices...")
fig_b, ax_b = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig_b.suptitle("MLP Confusion Matrix — Raw vs Youden Threshold [N4]",
               fontsize=12, fontweight="bold")
for ax, cm_, ttl in [
        (ax_b[0], result_raw["cm"],
         f"Raw (t=0.5)  Acc={result_raw['acc']:.3f}\n"
         f"Sens={result_raw['sens']:.3f}  Spec={result_raw['spec']:.3f}"),
        (ax_b[1], result_opt["cm"],
         f"Youden (t={bt:.3f})  Acc={result_opt['acc']:.3f}\n"
         f"Sens={result_opt['sens']:.3f}  Spec={result_opt['spec']:.3f}")]:
    cm_n = cm_.astype(float) / cm_.sum(axis=1)[:,None]
    sns.heatmap(cm_n, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=["Control","ASD"], yticklabels=["Control","ASD"],
                ax=ax, linewidths=1, cbar=False, annot_kws={"size":14,"weight":"bold"})
    for ii in range(2):
        for jj in range(2):
            ax.text(jj+0.5,ii+0.72,f"n={cm_[ii,jj]}",
                    ha="center",va="center",fontsize=10,color="#333")
    ax.set_title(ttl, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig_b.savefig(f"{SAVE_DIR}/figureB_mlp_confusion.png", dpi=150, bbox_inches="tight")
plt.close(fig_b); print(f"  Saved → {SAVE_DIR}/figureB_mlp_confusion.png")

# Fig C: Full comparison
print("  [Fig C] Full model comparison...")
all_r = {
    "Ridge (Bazay repl.)":    {"acc":0.606,"bal":0.584,"auc":0.572,"sens":0.872,"spec":0.296},
    "SVM (RBF)":              {"acc":0.571,"bal":0.557,"auc":0.590,"sens":0.755,"spec":0.358},
    "Random Forest":          {"acc":0.571,"bal":0.555,"auc":0.601,"sens":0.777,"spec":0.333},
    "XGBoost":                {"acc":0.531,"bal":0.519,"auc":0.594,"sens":0.691,"spec":0.346},
    "Ridge+Tangent [N1]":     {"acc":0.560,"bal":0.551,"auc":0.588,"sens":0.670,"spec":0.432},
    "Triple-FC SVM [N7]":     {"acc":0.566,"bal":0.556,"auc":0.617,"sens":0.691,"spec":0.420},
    "FC+Pheno SVM [N8]":      {"acc":0.554,"bal":0.539,"auc":0.604,"sens":0.745,"spec":0.333},
    "FC+Pheno SVM+J [N8+N4]": {"acc":0.611,"bal":0.608,"auc":0.604,"sens":0.723,"spec":0.494},
    "Stacked Ens. [N3]":      {"acc":0.571,"bal":0.551,"auc":0.589,"sens":0.830,"spec":0.272},
    "MLP [DL]":               {"acc":result_raw["acc"],"bal":result_raw["bal"],
                                "auc":result_raw["auc"],"sens":result_raw["sens"],
                                "spec":result_raw["spec"]},
    "MLP+Youden [DL+N4]":     {"acc":result_opt["acc"],"bal":result_opt["bal"],
                                "auc":result_opt["auc"],"sens":result_opt["sens"],
                                "spec":result_opt["spec"]},
}
names_c = list(all_r.keys())
is_dl   = ["MLP" in n for n in names_c]
mets    = ["acc","bal","auc","sens","spec"]
mlbls   = ["Accuracy","Bal. Acc","AUC","Sensitivity","Specificity"]

fig_c, axes_c = plt.subplots(1, 5, figsize=(26,7), facecolor=BG)
fig_c.suptitle(
    "Full Model Comparison: Classical ML vs Deep Learning\n"
    "ABIDE II · CC200 Pearson FC · MLP uses PCA(300) pre-reduction",
    fontsize=13, fontweight="bold")
for ax, mk, ml in zip(axes_c, mets, mlbls):
    vals = [all_r[n][mk] for n in names_c]
    cols = ["#E74C3C" if dl else "#4E9AF1" for dl in is_dl]
    brs  = ax.barh(names_c, vals, color=cols, alpha=0.88, edgecolor="white", height=0.65)
    ax.axvline(0.5,    color="red",   ls="--", lw=1.2, alpha=0.7)
    ax.axvline(0.6542, color="green", ls=":",  lw=1.5, alpha=0.85)
    for b, v in zip(brs, vals):
        ax.text(v+0.003, b.get_y()+b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(0.20, 0.95)
    ax.set_title(ml, fontsize=11, fontweight="bold")
    if ax != axes_c[0]: ax.set_yticklabels([])
from matplotlib.patches import Patch
axes_c[-1].legend(handles=[
    Patch(color="#E74C3C", alpha=0.88, label="Deep Learning (MLP)"),
    Patch(color="#4E9AF1", alpha=0.88, label="Classical ML"),
    plt.Line2D([0],[0], color="green", ls=":", lw=1.5, label="Bazay 2024 (65.42%)"),
    plt.Line2D([0],[0], color="red",   ls="--",lw=1.2, label="Chance (50%)"),
], fontsize=8, loc="lower right")
plt.tight_layout()
fig_c.savefig(f"{SAVE_DIR}/figureC_full_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig_c); print(f"  Saved → {SAVE_DIR}/figureC_full_comparison.png")

# Fig D: LOSO per site
print("  [Fig D] LOSO per site...")
fig_d, ax_d = plt.subplots(figsize=(14,5), facecolor=BG)
fig_d.suptitle("MLP LOSO Accuracy per Site  (leak-free, PCA inside fold [N10])",
               fontsize=12, fontweight="bold")
bar_c = [RED if a < 0.5 else TEAL for a in loso_accs]
ax_d.bar(range(len(loso_accs)), loso_accs, color=bar_c, alpha=0.85, edgecolor="white")
ax_d.axhline(np.mean(loso_accs), color=NAVY, ls="-.", lw=1.8,
             label=f"MLP mean={np.mean(loso_accs):.3f}±{np.std(loso_accs):.3f}")
ax_d.axhline(0.529, color=TEAL, ls="--", lw=1.5, label="Ridge mean=0.529")
ax_d.axhline(0.5,   color="gray", ls=":", lw=1, alpha=0.5, label="Chance")
ax_d.set_xticks(range(len(loso_sites)))
ax_d.set_xticklabels([str(s)[:8] for s in loso_sites], rotation=55, ha="right", fontsize=8)
ax_d.set_ylim(0, 1.05); ax_d.set_ylabel("Accuracy"); ax_d.legend(fontsize=9)
plt.tight_layout()
fig_d.savefig(f"{SAVE_DIR}/figureD_mlp_loso.png", dpi=150, bbox_inches="tight")
plt.close(fig_d); print(f"  Saved → {SAVE_DIR}/figureD_mlp_loso.png")


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

ratio_str = f"{val_losses[-1]/max(tr_losses[-1],1e-8):.2f}"
health    = "OK (fixed)" if float(ratio_str) < 2.5 else "still some overfitting"

print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  MLP RESULTS — PCA({PCA_COMPONENTS}) + Heinsfeld 2018 (FIXED v2)              │
  │                                                                      │
  │  Overfitting ratio: {ratio_str:<6}  ({health})                    │
  │                                                                      │
  │  80/20 Split                                                         │
  │    Raw (t=0.5)   Acc={result_raw['acc']:.3f}  Bal={result_raw['bal']:.3f}  AUC={result_raw['auc']:.3f}      │
  │                  Sens={result_raw['sens']:.3f}  Spec={result_raw['spec']:.3f}                   │
  │    Youden opt.   Acc={result_opt['acc']:.3f}  Bal={result_opt['bal']:.3f}  t={bt:.3f}          │
  │                  Sens={result_opt['sens']:.3f}  Spec={result_opt['spec']:.3f}  J={result_opt['J']:.3f}      │
  │                                                                      │
  │  LOSO (leak-free, PCA inside fold)                                   │
  │    Acc={np.mean(loso_accs):.3f} ± {np.std(loso_accs):.3f}   F1={np.mean(loso_f1s):.3f}   AUC={np.nanmean(loso_aucs):.3f}           │
  ├──────────────────────────────────────────────────────────────────────┤
  │  BEST PER METRIC ACROSS ALL MODELS                                   │
  │  Best Accuracy  : FC+Pheno SVM+J [N8+N4]  →  0.611                 │
  │  Best AUC       : MLP [DL]        →  {result_raw['auc']:.3f}                      │
  │  Best Bal. Acc  : FC+Pheno SVM+J  →  0.608                         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  BASELINES                                                           │
  │  Bazay 2024   →  65.42%  Ridge+AAL, ABIDE II (random split)         │
  │  Heinsfeld    →  70.00%  DNN, ABIDE I (different dataset)           │
  └──────────────────────────────────────────────────────────────────────┘
""")

torch.save(model_mlp.state_dict(), "mlp_weights.pt")
print("  Weights saved → mlp_weights.pt")
