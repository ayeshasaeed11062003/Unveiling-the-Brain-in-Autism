import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
import random

# ==========================================================
# Reproducibility
# ==========================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")

# ==========================================================
# Load Data (YOUR EXACT PATHS)
# ==========================================================
X = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\graph_embeddings.npy")
y_dx = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\dx_labels.npy")        # 1=ASD, 2=Control
y_site_raw = np.load("c:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\site_labels.npy")  # Site labels

# Convert dx to binary 0/1
y_dx = np.where(y_dx == 1, 1, 0)

# Encode site labels safely
site_encoder = LabelEncoder()
y_site = site_encoder.fit_transform(y_site_raw)

print("Data Loaded")
print("Original X shape:", X.shape)
print("Unique sites:", len(np.unique(y_site)))
print("-----------------------------------")

# ==========================================================
# PCA (CRITICAL for stability)
# ==========================================================
print("Applying PCA to 500 components...")
pca = PCA(n_components=500, random_state=42)
X = pca.fit_transform(X)
print("After PCA shape:", X.shape)
print("-----------------------------------")

# ==========================================================
# Dataset
# ==========================================================
class BrainDataset(Dataset):
    def __init__(self, X, y_dx, y_site):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_dx = torch.tensor(y_dx, dtype=torch.long)
        self.y_site = torch.tensor(y_site, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_dx[idx], self.y_site[idx]

# ==========================================================
# Gradient Reversal Layer
# ==========================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# ==========================================================
# DANN Model
# ==========================================================
class DANN(nn.Module):
    def __init__(self, input_dim, num_sites):
        super(DANN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.dx_classifier = nn.Linear(128, 2)
        self.domain_classifier = nn.Linear(128, num_sites)

    def forward(self, x, lambda_):
        features = self.feature_extractor(x)
        dx_out = self.dx_classifier(features)

        reversed_features = GradReverse.apply(features, lambda_)
        domain_out = self.domain_classifier(reversed_features)

        return dx_out, domain_out, features

# ==========================================================
# Contrastive Loss (Stable Version)
# ==========================================================
def contrastive_loss(features, labels, temperature=0.5):
    features = nn.functional.normalize(features, dim=1)
    similarity = torch.matmul(features, features.T)

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    logits = similarity / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0]

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    return -mean_log_prob_pos.mean()

# ==========================================================
# Train One LOSO Fold
# ==========================================================
def train_one_fold(X_train, y_dx_train, y_site_train,
                   X_test, y_dx_test, y_site_test,
                   num_sites):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = BrainDataset(X_train, y_dx_train, y_site_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = DANN(X_train.shape[1], num_sites).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dx_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    epochs = 30
    total_steps = epochs * len(train_loader)
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss_epoch = 0

        for x_batch, dx_batch, site_batch in train_loader:

            x_batch = x_batch.to(device)
            dx_batch = dx_batch.to(device)
            site_batch = site_batch.to(device)

            # Slower lambda ramp
            p = float(step) / total_steps
            lambda_domain = 0.1 * (2. / (1. + np.exp(-5 * p)) - 1)

            dx_out, domain_out, features = model(x_batch, lambda_domain)

            dx_loss = dx_criterion(dx_out, dx_batch)
            domain_loss = domain_criterion(domain_out, site_batch)
            cont_loss = contrastive_loss(features, dx_batch)

            # Balanced objective
            loss = dx_loss + 0.05 * domain_loss + 0.02 * cont_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            step += 1

        print(f"Epoch {epoch+1} | Loss: {total_loss_epoch:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        dx_out, domain_out, _ = model(X_test_tensor, lambda_=0)

        dx_preds = torch.argmax(dx_out, dim=1).cpu().numpy()
        site_preds = torch.argmax(domain_out, dim=1).cpu().numpy()

    asd_acc = accuracy_score(y_dx_test, dx_preds)
    asd_f1 = f1_score(y_dx_test, dx_preds)
    site_acc = accuracy_score(y_site_test, site_preds)

    return asd_acc, asd_f1, site_acc

# ==========================================================
# LOSO Cross Validation
# ==========================================================
logo = LeaveOneGroupOut()

asd_accs = []
asd_f1s = []
site_accs = []

for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y_dx, groups=y_site)):

    print("\n===================================")
    print(f"LOSO Fold {fold+1}")
    print("===================================")

    X_train, X_test = X[train_idx], X[test_idx]
    y_dx_train, y_dx_test = y_dx[train_idx], y_dx[test_idx]
    y_site_train, y_site_test = y_site[train_idx], y_site[test_idx]

    acc, f1, s_acc = train_one_fold(
        X_train, y_dx_train, y_site_train,
        X_test, y_dx_test, y_site_test,
        num_sites=len(np.unique(y_site))
    )

    print("ASD Accuracy:", acc)
    print("ASD F1:", f1)
    print("Site Accuracy:", s_acc)

    asd_accs.append(acc)
    asd_f1s.append(f1)
    site_accs.append(s_acc)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")
print("Mean ASD Accuracy:", np.mean(asd_accs))
print("Mean ASD F1:", np.mean(asd_f1s))
print("Mean Site Accuracy:", np.mean(site_accs))