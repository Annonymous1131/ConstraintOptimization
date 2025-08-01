import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import pandas as pd
import sys

from Networks import TabularMLP, TabularMLP_Mult
from RunGurobi import FlipBinary_A, FlipBinary_C, FlipMulticlass_A, FlipMulticlass_C, BorderBinary, BorderMulticlass


def LoadDataset(name="adult", run_id=0):
    if name == "santander":
        df = pd.read_csv(f"data/{name}/train.csv")
        target_col = "target"
    else:
        dataset = fetch_openml(name, version=1, as_frame=True)
        df = dataset.frame.dropna()
        target_col = dataset.target_names[0] if hasattr(dataset, 'target_names') else dataset.target
    
    y = df[target_col]
    X = df.drop(columns=[target_col])

    y = LabelEncoder().fit_transform(y)

    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    num_cols = X.select_dtypes(include=["float", "int"]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y.astype(np.float32 if len(np.unique(y)) == 2 else np.int64),
        test_size=0.2,
        random_state=42*run_id,
        stratify=y
    )

    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    num_classes = len(np.unique(y_train.numpy()))

    return X_train, y_train, X_test, y_test, num_classes


def accuracy_binary(logits, labels):
    preds = torch.sigmoid(logits).round()
    return (preds == labels).float().mean().item()

def accuracy_multiclass(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def TrainNN(Dataset, X_train, y_train, X_test, y_test, num_classes=2, patience=3, max_epochs=100, preset_weights_path=None, run_id=0, Method=None):
    IntermediateStatsFile = f"Stats/Intermediate/{Dataset}_{run_id}_Epochs.csv"
    AllFileStats = f"Stats/Summary.csv"
    os.makedirs(os.path.dirname(IntermediateStatsFile), exist_ok=True)
    if os.path.exists(IntermediateStatsFile) == False:
        with open(IntermediateStatsFile, "w") as f:
            f.write("Epoch,Train Loss,Train Acc,Test Acc,Method\n")
    if os.path.exists(AllFileStats) == False:
        with open(AllFileStats, "w") as f:
            f.write("Dataset,Run,Train Loss,Train Acc,Test Acc,Method\n")
    

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{Dataset}_{run_id}_tabular_model_{Method}.pt"
    prediction_path = f"checkpoints/{Dataset}_{run_id}_tabular_preds_{Method}.npy"
    GT_path = f"checkpoints/{Dataset}_{run_id}_tabular_gt_{Method}.npy"

    input_dim = X_train.shape[1]

    if num_classes == 2:
        model = TabularMLP(input_dim=input_dim)
        criterion = nn.BCEWithLogitsLoss()
    else:
        model = TabularMLP_Mult(input_dim=input_dim, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if preset_weights_path is not None and os.path.isfile(preset_weights_path):
        model.load_state_dict(torch.load(preset_weights_path))

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        logits = model(X_train)
        if num_classes == 2:
            logits = logits.squeeze()
            loss = criterion(logits, y_train)
        else:
            loss = criterion(logits, y_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            test_logits = model(X_test)

            if num_classes == 2:
                train_acc = accuracy_binary(train_logits.squeeze(), y_train)
                test_acc = accuracy_binary(test_logits.squeeze(), y_test)
            else:
                train_acc = accuracy_multiclass(train_logits, y_train)
                test_acc = accuracy_multiclass(test_logits, y_test)

        print(f"Epoch {epoch+1} | Train Loss: {loss.item():.8f} | Train Acc: {train_acc:.8f} | Test Acc: {test_acc:.8f}")
        with open(IntermediateStatsFile, "a") as f:
            f.write(f"{epoch+1},{loss.item():.8f},{train_acc:.8f},{test_acc:.8f},{Method}\n")
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered (train loss did not improve).")
                break

    with open(AllFileStats, "a") as f:
        f.write(f"{Dataset},{run_id},{loss.item():.8f},{train_acc:.8f},{test_acc:.8f},{Method}\n")
            
    
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    with torch.no_grad():
        final_train_logits = model(X_train)
        if num_classes == 2:
            preds = torch.sigmoid(final_train_logits).round().cpu().numpy()
        else:
            preds = torch.argmax(final_train_logits, dim=1).cpu().numpy()

        gt = y_train.cpu().numpy()

    np.save(prediction_path, preds)
    np.save(GT_path, gt)

def evaluate_tensors(model, X, y, num_classes):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if num_classes == 2:
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(logits.view(-1), y.float())
            preds = torch.sigmoid(logits).round().view(-1)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, y.long())
            preds = torch.argmax(logits, dim=1)

        accuracy = (preds == y.view(-1)).float().mean().item()

    return loss.item(), accuracy, preds.cpu().numpy()


def ModifyWeights(Dataset, X_train, y_train, X_test, y_test, num_classes=2, n_samples=1000, flipCount=1, tol=1e-5, run_id=0, Method="F"):
    checkpoint_path = f"checkpoints/{Dataset}_{run_id}_tabular_model_Train.pt"
    prediction_path = f"checkpoints/{Dataset}_{run_id}_tabular_preds_Train.npy"
    GT_path = f"checkpoints/{Dataset}_{run_id}_tabular_gt_Train.npy"

    G_checkpoint_path = f"checkpoints/{Dataset}_{run_id}_tabular_model_{Method}.pt"

    RAG_checkpoint_path = f"checkpoints/{Dataset}_{run_id}_tabular_model_RA{Method}.pt"
    RAG_prediction_path = f"checkpoints/{Dataset}_{run_id}_tabular_preds_RA{Method}.npy"
    RAG_GT_path = f"checkpoints/{Dataset}_{run_id}_tabular_gt_RA{Method}.npy"

    AllFileStats = f"Stats/Summary.csv"

    input_dim = X_train.shape[1]
    if num_classes == 2:
        model = TabularMLP(input_dim=input_dim)
    else:
        model = TabularMLP_Mult(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    with torch.no_grad():
        model(X_train, store_A_last=True)
        A_last = model.A_last_cache
        W = model.final_layer.weight.detach().clone()
        b = model.final_layer.bias.detach().clone()

    y_gt = np.load(GT_path, allow_pickle=True)
    y_preds = np.load(prediction_path, allow_pickle=True)
    if Method == "F_A":
        if num_classes == 2:            
            G_result = FlipBinary_A(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol, flipCount)
        else:
            G_result = FlipMulticlass_A(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol, flipCount)
    elif Method == "F_C":
        if num_classes == 2:
            G_result = FlipBinary_C(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol, flipCount)
        else:
            G_result = FlipMulticlass_C(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol, flipCount)

    elif Method == "B":
        if num_classes == 2:
            G_result = BorderBinary(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol)
        else:
            G_result = BorderMulticlass(Dataset, A_last, y_preds, y_gt, W, b, n_samples, tol)
    
    if G_result is not None:
        W_new, b_new = G_result
    else:
        return False
    W_new = torch.tensor(W_new, dtype=torch.float32)
    b_new = torch.tensor(b_new, dtype=torch.float32)
    
    with torch.no_grad():
        model.final_layer.weight.copy_(W_new)
        model.final_layer.bias.copy_(b_new)

    train_loss, train_acc, gurobi_train_preds = evaluate_tensors(model, X_train, y_train, num_classes)
    test_loss, test_acc, gurobi_test_preds = evaluate_tensors(model, X_test, y_test, num_classes)

    with open(AllFileStats, "a") as f:
        f.write(f"{Dataset},{run_id},{train_loss:.8f},{train_acc:.8f},{test_acc:.8f},{Method}{flipCount}\n")
    
    Total_flips = np.sum(gurobi_train_preds != y_preds.reshape(-1)).item()
    with open(f"Stats/Log.txt", "a") as f:
        f.write(f"Dataset: {Dataset}, Method: {Method}{flipCount}, Run ID: {run_id}, Total Flips: {Total_flips}, Expected Flips: {flipCount}\n")
    if Total_flips != flipCount:
        print(f"Warning: Expected {flipCount} flips, but found {Total_flips} flips after Gurobi optimization.")

    torch.save(model.state_dict(), G_checkpoint_path)

    TrainNN(Dataset, X_train, y_train, X_test, y_test, num_classes=num_classes, patience=15, max_epochs=200000, preset_weights_path=G_checkpoint_path, run_id=run_id, Method=f"RA{Method}{flipCount}")
    
    return True

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "STC"
    misc_type = sys.argv[3] if len(sys.argv) > 3 else "A"
    misclassification_count = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    # Datasets = ["Adult", "higgs", "GiveMeSomeCredit", "bank-marketing", "santander"]

    X_train, y_train, X_test, y_test, num_classes = LoadDataset(dataset_name, run_id=1)
    
    TrainNN(dataset_name, X_train, y_train, X_test, y_test, num_classes=num_classes, patience=15, max_epochs=200000, preset_weights_path=None, run_id=1, Method="Train")
    
    if method == "CmC":
        if misc_type == "A":
            Solution = ModifyWeights(dataset_name, X_train, y_train, X_test, y_test, num_classes=num_classes, n_samples=1000, flipCount=misclassification_count, tol=1e-5, run_id=1, Method="F_A")
        elif misc_type == "C":
            Solution = ModifyWeights(dataset_name, X_train, y_train, X_test, y_test, num_classes=num_classes, n_samples=1000, flipCount=misclassification_count, tol=1e-5, run_id=1, Method="F_C")

    elif method == "STC":
        Solution = ModifyWeights(dataset_name, X_train, y_train, X_test, y_test, num_classes=num_classes, n_samples=-1, flipCount=0, tol=1e-5, run_id=1, Method="B")

    if Solution:
        if method == "CmC":
            method_suffix = f"F_{misc_type}"
        else:
            method_suffix = "B"

        print(f"Successfully modified weights for dataset \"{dataset_name}\" using method \"{method}\".")
        print(f"Path to model_G: checkpoints/{dataset_name}_1_tabular_model_{method_suffix}.pt")
        if method == "CmC":
            print(f"Path to model_RT (Retrain after misclassification): checkpoints/{dataset_name}_1_tabular_model_RA{method_suffix}{misclassification_count}.pt")

    else:
        print(f"Failed to modify weights for dataset \"{dataset_name}\" using method \"{method}\".")

