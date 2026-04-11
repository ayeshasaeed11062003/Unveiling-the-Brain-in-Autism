"""
baseline_models.py  —  Unveiling the Brain in Autism
Fix: padded_matrices_200.npy contains raw covariance values (not correlations).
We convert to correlation matrices first, then extract upper triangle FC features.
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection   import train_test_split, LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.decomposition     import PCA
from sklearn.linear_model      import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.ensemble          import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics           import (accuracy_score, f1_score,
                                       confusion_matrix, roc_curve, roc_auc_score)
from sklearn.pipeline          import Pipeline
from sklearn.svm               import SVC
from sklearn.manifold          import TSNE
from sklearn.calibration       import CalibratedClassifierCV
from xgboost                   import XGBClassifier

SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

PALETTE = {
    "Ridge":           "#4E9AF1",
    "SVM (RBF)":       "#9B59B6",
    "Random Forest":   "#2EC4B6",
    "XGBoost":         "#F39C12",
    "Logistic Reg.":   "#7BC8F6",
    "Voting Ensemble": "#E84545",
}
BG = "#F4F6F8"; RED = "#E84545"; TEAL = "#2B7A78"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

def section(t):
    print(f"\n{'='*62}\n  {t}\n{'='*62}")


def cov_to_corr(M):
    d = np.sqrt(np.abs(np.diag(M)))
    d[d < 1e-8] = 1e-8
    return M / np.outer(d, d)


# ── 1. LOAD & CONVERT ────────────────────────────────────────
section("1 / 5   LOADING & CONVERTING TO CORRELATION")

matrices = np.load("padded_matrices_200.npy")   # (871, 200, 200) covariance
y_raw    = np.load("dx_labels.npy")
y_site   = np.load("site_labels.npy")
y_dx     = (y_raw - 1).astype(int)

print(f"  Raw matrix range: {matrices.min():.2f} to {matrices.max():.2f}  (covariance)")
print(f"  Converting {len(matrices)} subjects to correlation matrices...")

corr_matrices = np.array([cov_to_corr(m) for m in matrices])
corr_matrices = np.clip(corr_matrices, -1.0, 1.0)
corr_matrices = np.nan_to_num(corr_matrices, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(corr_matrices)):
    np.fill_diagonal(corr_matrices[i], 0)

print(f"  Correlation range: {corr_matrices.min():.3f} to {corr_matrices.max():.3f}  (correct)")

# Upper triangle -> FC features
mask = np.triu(np.ones((200, 200), dtype=bool), k=1)
X    = np.array([c[mask] for c in corr_matrices])   # (871, 19900)

# Fisher z-transform
X = np.arctanh(np.clip(X, -0.9999, 0.9999))

print(f"  FC features : {X.shape}")
print(f"  Classes     : {dict(zip(*np.unique(y_dx, return_counts=True)))}")
print(f"  Sites       : {len(np.unique(y_site))}")


# ── 2. SITE RESIDUALISATION ──────────────────────────────────
section("2 / 5   SITE RESIDUALISATION")

def residualize(X, site_labels):
    enc = OneHotEncoder(sparse_output=False)
    S   = enc.fit_transform(np.array(site_labels).reshape(-1, 1))
    X_r = X.copy().astype(float)
    for i in range(X.shape[1]):
        reg       = LinearRegression().fit(S, X_r[:, i])
        X_r[:, i] = X_r[:, i] - reg.predict(S)
    return X_r

print("  Residualising...", end=" ", flush=True)
X_res = residualize(X, y_site)
print("done.")

site_arr       = np.array(y_site)
idx            = np.arange(len(y_dx))
idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_dx)
X_tr, X_te     = X_res[idx_tr], X_res[idx_te]
y_tr, y_te     = y_dx[idx_tr],  y_dx[idx_te]
print(f"  Train: {len(y_tr)}   Test: {len(y_te)}")

K          = 2000
pos_weight = float(np.sum(y_dx==0)) / float(np.sum(y_dx==1))


# ── 3. TRAIN MODELS ──────────────────────────────────────────
section("3 / 5   TRAINING MODELS")

# Tune Ridge alpha
best_alpha, best_cv = 1.0, 0.0
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=K)),
        ("clf", RidgeClassifier(alpha=alpha, class_weight="balanced")),
    ])
    sc = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy").mean()
    if sc > best_cv:
        best_cv, best_alpha = sc, alpha
print(f"  Ridge best alpha={best_alpha}  CV={best_cv:.3f}")

base_models = {
    "Ridge": Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=K)),
        ("clf", CalibratedClassifierCV(
                    RidgeClassifier(alpha=best_alpha, class_weight="balanced"), cv=5)),
    ]),
    "SVM (RBF)": Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=K)),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=K)),
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=10,
                    min_samples_leaf=2, class_weight="balanced",
                    random_state=42, n_jobs=1)),
    ]),
    "XGBoost": Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=500)),
        ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                    scale_pos_weight=pos_weight, eval_metric="logloss",
                    random_state=42, verbosity=0, nthread=1)),
    ]),
    "Logistic Reg.": Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=K)),
        ("pca", PCA(n_components=150)),
        ("clf", LogisticRegression(max_iter=5000, C=0.1,
                    class_weight="balanced", solver="lbfgs")),
    ]),
}
base_models["Voting Ensemble"] = VotingClassifier(
    estimators=[("r", base_models["Ridge"]),
                ("s", base_models["SVM (RBF)"]),
                ("x", base_models["XGBoost"])],
    voting="soft", n_jobs=1)

results = {}
for name, mdl in base_models.items():
    print(f"  {name}...", end=" ", flush=True)
    mdl.fit(X_tr, y_tr)
    yp  = mdl.predict(X_te)
    ypr = mdl.predict_proba(X_te)[:,1] if hasattr(mdl,"predict_proba") else None
    cm  = confusion_matrix(y_te, yp)
    tn,fp,fn,tp = cm.ravel()
    results[name] = dict(
        model=mdl, y_pred=yp, y_prob=ypr,
        acc=accuracy_score(y_te,yp),
        f1=f1_score(y_te,yp,average="weighted"),
        auc=roc_auc_score(y_te,ypr) if ypr is not None else np.nan,
        cm=cm, sens=tp/max(tp+fn,1), spec=tn/max(tn+fp,1))
    r = results[name]
    print(f"Acc={r['acc']:.3f}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}")

names     = list(results.keys())
colors    = [PALETTE[n] for n in names]
best_name = max(results, key=lambda n: results[n]["acc"])
print(f"\n  BEST: {best_name}  Acc={results[best_name]['acc']:.3f}")
print(f"  Bazay 2024 target: 0.6542")
if results[best_name]['acc'] >= 0.6542:
    print("  *** MATCHES OR BEATS BAZAY 2024 ***")
else:
    print(f"  Gap to target: {0.6542 - results[best_name]['acc']:.3f}")


# ── 4. LOSO CV ───────────────────────────────────────────────
section("4 / 5   LEAVE-ONE-SITE-OUT CV")

loso_accs, loso_f1s, loso_sites = [], [], []
for tr_i, te_i in LeaveOneGroupOut().split(X_res, y_dx, groups=site_arr):
    m = Pipeline([
        ("sc",  StandardScaler()),
        ("sel", SelectKBest(f_classif, k=500)),
        ("clf", CalibratedClassifierCV(
                    RidgeClassifier(alpha=best_alpha, class_weight="balanced"), cv=3)),
    ])
    m.fit(X_res[tr_i], y_dx[tr_i])
    yp = m.predict(X_res[te_i])
    loso_accs.append(accuracy_score(y_dx[te_i], yp))
    loso_f1s.append(f1_score(y_dx[te_i], yp, average="weighted"))
    loso_sites.append(np.unique(site_arr[te_i])[0])

print(f"  LOSO Accuracy : {np.mean(loso_accs):.4f}")
print(f"  LOSO F1       : {np.mean(loso_f1s):.4f}")


# ── 5. FIGURES ───────────────────────────────────────────────
section("5 / 5   FIGURES")

print("  [Fig 1] Dashboard...")
fig1 = plt.figure(figsize=(22,16), facecolor=BG)
fig1.suptitle("Unveiling the Brain in Autism  --  Model Performance Dashboard\n"
              "ABIDE II  --  CC200 Correlation FC  --  Site-Residualised",
              fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3,3,figure=fig1,hspace=0.50,wspace=0.38)

for col,(mk,ml) in enumerate([("acc","Accuracy"),("f1","Weighted F1"),("auc","ROC-AUC")]):
    ax  = fig1.add_subplot(gs[0,col])
    vs  = [results[n][mk] for n in names]
    brs = ax.barh(names, vs, color=colors, alpha=0.88, edgecolor="white", height=0.55)
    ax.axvline(0.5,    color="red",   ls="--", lw=1.3, alpha=0.7, label="Chance")
    ax.axvline(0.6542, color="green", ls=":",  lw=1.8, alpha=0.9, label="Bazay 2024")
    if mk=="acc": ax.legend(fontsize=8)
    for b,v in zip(brs,vs):
        ax.text(v+0.005, b.get_y()+b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0.35,0.85); ax.set_title(ml,fontsize=12,fontweight="bold")
    ax.set_xlabel("Score",fontsize=9)
    if col>0: ax.set_yticklabels([])

ax = fig1.add_subplot(gs[1,0])
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
for name in names:
    r = results[name]
    if r["y_prob"] is not None:
        fpr,tpr,_ = roc_curve(y_te,r["y_prob"])
        ax.plot(fpr,tpr,lw=2.5 if name==best_name else 1.5,
                color=PALETTE[name],label=f"{name} ({r['auc']:.3f})")
ax.set_xlim(0,1); ax.set_ylim(0,1.02)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves",fontsize=12,fontweight="bold")
ax.legend(fontsize=7.5,loc="lower right")

ax = fig1.add_subplot(gs[1,1])
for name in names:
    r = results[name]
    ax.scatter(r["spec"],r["sens"],s=140,color=PALETTE[name],
               zorder=5,edgecolors="white",lw=1.5)
    ax.annotate(name.split()[0],(r["spec"]+0.01,r["sens"]+0.01),
                fontsize=8,color=PALETTE[name],fontweight="bold")
ax.axhline(0.5,color="gray",ls="--",lw=1,alpha=0.5)
ax.axvline(0.5,color="gray",ls="--",lw=1,alpha=0.5)
ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("Sensitivity vs Specificity",fontsize=12,fontweight="bold")
ax.set_xlim(0.1,1.0); ax.set_ylim(0.1,1.0)

ax  = fig1.add_subplot(gs[1,2])
mks = ["acc","f1","auc","sens","spec"]
mls = ["Acc","F1","AUC","Sens","Spec"]
mc  = ["#4E9AF1","#2EC4B6","#F39C12",RED,TEAL]
x_  = np.arange(len(names)); bw=0.15
for i,(mk,ml) in enumerate(zip(mks,mls)):
    ax.bar(x_+i*bw,[results[n][mk] for n in names],bw,
           label=ml,color=mc[i],alpha=0.85,edgecolor="white")
ax.set_xticks(x_+bw*2)
ax.set_xticklabels([n.split()[0] for n in names],rotation=30,ha="right",fontsize=8)
ax.axhline(0.5,color="red",ls="--",lw=1,alpha=0.5)
ax.set_ylim(0,1); ax.set_title("All Metrics",fontsize=12,fontweight="bold")
ax.legend(fontsize=8,loc="upper left")

ax   = fig1.add_subplot(gs[2,0])
cm_b = results[best_name]["cm"]
cm_n = cm_b.astype(float)/cm_b.sum(axis=1)[:,None]
sns.heatmap(cm_n,annot=True,fmt=".1%",cmap="Blues",
            xticklabels=["Control","ASD"],yticklabels=["Control","ASD"],
            ax=ax,linewidths=1,cbar=False,annot_kws={"size":13,"weight":"bold"})
for ii in range(2):
    for jj in range(2):
        ax.text(jj+0.5,ii+0.72,f"n={cm_b[ii,jj]}",
                ha="center",va="center",fontsize=9,color="#333")
ax.set_title(f"Best: {best_name}\nAcc={results[best_name]['acc']:.3f}",
             fontsize=11,fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")

ax = fig1.add_subplot(gs[2,1])
ax.bar(range(len(loso_accs)),loso_accs,
       color=[RED if a<0.5 else TEAL for a in loso_accs],alpha=0.85,edgecolor="white")
ax.axhline(0.5,color="red",ls="--",lw=1.5,label="Chance")
ax.axhline(np.mean(loso_accs),color="navy",ls="-.",lw=1.5,
           label=f"Mean={np.mean(loso_accs):.3f}")
ax.set_xticks(range(len(loso_sites)))
ax.set_xticklabels([str(s)[:7] for s in loso_sites],rotation=55,ha="right",fontsize=7)
ax.set_ylim(0,1); ax.set_title("LOSO per Site",fontsize=11,fontweight="bold")
ax.set_ylabel("Accuracy"); ax.legend(fontsize=9)

ax = fig1.add_subplot(gs[2,2])
lit_n = ["Chance","Our LOSO\n(Ridge)",f"Our Best\n({best_name.split()[0]})",
         "Bazay 2024\n(Ridge+AAL)","Heinsfeld\n2018 (DNN)"]
lit_v = [0.50,np.mean(loso_accs),results[best_name]["acc"],0.6542,0.70]
lit_c = ["#aaa",TEAL,RED,"#F39C12","#9B59B6"]
bars  = ax.bar(lit_n,lit_v,color=lit_c,alpha=0.88,edgecolor="white")
for b,v in zip(bars,lit_v):
    ax.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",
            ha="center",fontsize=9,fontweight="bold")
ax.set_ylim(0,0.85); ax.set_title("Literature Comparison",fontsize=11,fontweight="bold")
ax.set_ylabel("Accuracy"); ax.tick_params(axis="x",labelsize=7)
ax.axhline(0.6542,color="green",ls=":",lw=1.5,alpha=0.7)

fig1.savefig(f"{SAVE_DIR}/figure1_dashboard.png",dpi=150,bbox_inches="tight")
plt.close(fig1)
print(f"       Saved -> {SAVE_DIR}/figure1_dashboard.png")

print("  [Fig 2] Confusion matrices...")
fig2,axes2 = plt.subplots(2,3,figsize=(16,10),facecolor=BG)
fig2.suptitle("Confusion Matrices -- All 6 Models  (ABIDE II -- CC200 Corr FC -- site-residualised)",
              fontsize=14,fontweight="bold",y=1.01)
for ax,name in zip(axes2.flat,names):
    cm_ = results[name]["cm"]
    cm_n = cm_.astype(float)/cm_.sum(axis=1)[:,None]
    sns.heatmap(cm_n,annot=True,fmt=".1%",cmap="RdBu_r",center=0.5,
                xticklabels=["Control","ASD"],yticklabels=["Control","ASD"],
                ax=ax,linewidths=1,cbar=False,annot_kws={"size":12,"weight":"bold"})
    for ii in range(2):
        for jj in range(2):
            ax.text(jj+0.5,ii+0.72,f"n={cm_[ii,jj]}",
                    ha="center",va="center",fontsize=9,color="#333")
    ax.set_title(f"{name}\nAcc={results[name]['acc']:.3f}  "
                 f"F1={results[name]['f1']:.3f}  AUC={results[name]['auc']:.3f}",
                 fontsize=10,fontweight="bold",color=PALETTE[name])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
fig2.savefig(f"{SAVE_DIR}/figure2_confusion_matrices.png",dpi=150,bbox_inches="tight")
plt.close(fig2)
print(f"       Saved -> {SAVE_DIR}/figure2_confusion_matrices.png")

print("  [Fig 3] Embeddings...")
Xv   = StandardScaler().fit_transform(X_res)
Xv50 = PCA(n_components=50,random_state=42).fit_transform(Xv)
X_pca = Xv50[:,:2]
print("  t-SNE...", end=" ", flush=True)
X_tsne = TSNE(n_components=2,perplexity=30,random_state=42,
              init="pca",max_iter=1000).fit_transform(Xv50)
print("done.")

fig3,ax3 = plt.subplots(1,3,figsize=(19,6),facecolor=BG)
fig3.suptitle("CC200 Correlation FC Embedding  --  ABIDE II Domain Shift",
              fontsize=13,fontweight="bold")
for ax,Xv2,ttl,cb in [(ax3[0],X_pca,"PCA by Diagnosis","dx"),
                       (ax3[1],X_tsne,"t-SNE by Diagnosis","dx"),
                       (ax3[2],X_tsne,"t-SNE by Site","site")]:
    if cb=="dx":
        for lbl,c,lab in [(0,TEAL,"Control"),(1,RED,"ASD")]:
            m = y_dx==lbl
            ax.scatter(Xv2[m,0],Xv2[m,1],c=c,alpha=0.5,s=15,label=lab,edgecolors="none")
        ax.legend(fontsize=10)
    else:
        us  = np.unique(site_arr)
        cmp = plt.cm.tab20(np.linspace(0,1,len(us)))
        for i,s in enumerate(us):
            m = site_arr==s
            ax.scatter(Xv2[m,0],Xv2[m,1],color=cmp[i],alpha=0.65,
                       s=15,label=str(s)[:8],edgecolors="none")
        ax.legend(fontsize=6,ncol=2,loc="lower right",title="Site",title_fontsize=7)
    ax.set_title(ttl,fontsize=11,fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
fig3.savefig(f"{SAVE_DIR}/figure3_embeddings.png",dpi=150,bbox_inches="tight")
plt.close(fig3)
print(f"       Saved -> {SAVE_DIR}/figure3_embeddings.png")


section("RESULTS SUMMARY")
print(f"\n  {'Model':<22} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Sens':>7} {'Spec':>7}")
print(f"  {'-'*62}")
for name in names:
    r = results[name]
    star = "  <- BEST" if name==best_name else ""
    print(f"  {name:<22} {r['acc']:>7.3f} {r['f1']:>7.3f} "
          f"{r['auc']:>7.3f} {r['sens']:>7.3f} {r['spec']:>7.3f}{star}")
print(f"  {'-'*62}")
print(f"  {'LOSO (Ridge)':<22} {np.mean(loso_accs):>7.3f} "
      f"{np.mean(loso_f1s):>7.3f}   (cross-site holdout)")
print(f"\n  TARGET: Bazay 2024 -> 65.42%  (Ridge + AAL, ABIDE II)")
print(f"  KEY FIX: Converted covariance -> correlation matrices before FC extraction")
