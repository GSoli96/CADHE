import copy
import csv
import json
import os
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tenseal as ts
import torch
from debugpy.launcher.debuggee import describe
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from CNN.convNet import ConvNet
from CNN.encConvNet import EncConvNet


# Callback EarlyStopping
class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.epoch = epoch

class train_test_ConvNet():
    def __init__(self,train_loader, val_loader, test_loader,hidden, hidden2, lr, batch_size, image_size_general, grid_path, num_classes, epochs, position):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.hidden = hidden
        self.hidden2 = hidden2
        self.lr = lr
        self.batch_size = batch_size
        self.grid_path = grid_path
        self.image_size_general = image_size_general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopper = EarlyStopping(patience=4)
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.history = []
        self.epochs = epochs
        self.num_classes = num_classes
        self.model = ConvNet(image_size_general_local=self.image_size_general, hidden=self.hidden, hidden2=self.hidden2, num_classes = self.num_classes).to(self.device)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = 0.0
        self.model_path = os.path.join(self.grid_path, f"model_h{self.hidden}_h2{self.hidden2}_lr{self.lr}_bs{self.batch_size}.pt")
        self.position = position
        self.train_ConvNet()

    def train_ConvNet(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        for epoch in tqdm(range(1, self.epochs), desc=f'Training ConvNet ({self.grid_path.split("/")[-1]})', total=self.epochs, colour='green', leave=False):  # max 50 epoche
            self.model.train()
            train_loss = 0.0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # Validazione
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)
            val_acc = correct / total

            scheduler.step(val_loss)
            self.history.append(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            # Checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.model_path)

            self.early_stopper(val_loss, epoch)
            if self.early_stopper.early_stop:
                self.history.append(f"Early stopping at epoch {epoch}")
                break

    def test_ConvNet(self):
        # Salvataggio curva di loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Loss - h={self.hidden}, h2={self.hidden2}, lr={self.lr}, bs={self.batch_size}")
        plt.legend()
        plot_path = os.path.join(self.grid_path, f"loss_h{self.hidden}_h2{self.hidden2}_lr{self.lr}_bs{self.batch_size}.pdf")
        plt.savefig(plot_path)
        plt.close()

        tmp = {
            'hidden': self.hidden,
            'hidden2': self.hidden2,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'val_acc': self.best_val_acc,
            'loss_plot': plot_path,
            'early_stopper': self.early_stopper.epoch,
            'model_path': self.model_path
        }

        # ---------------------- TEST EVALUATION ----------------------
        self.model.load_state_dict(self.best_model_wts)
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_cnn(num_classes=self.num_classes,
                 all_preds=all_preds,
                 all_targets=all_targets,
                 tmp=tmp,
                 history = self.history,
                 out_put=os.path.join(self.grid_path,
                                   f"report_h{self.hidden}_h2{self.hidden2}_lr{self.lr:.0e}_bs{self.batch_size}.txt"),
                 pdf_path=os.path.join(self.grid_path,
                                        f"confusion_h{self.hidden}_h2{self.hidden2}_lr{self.lr:.0e}_bs{self.batch_size}.pdf"),
                 model_save_path = None,
                 model=None,
                 overall_res=None
                 )

        return tmp

class train_test_EncConvNet():
    def __init__(self, dataset, num_classes, seed=None, index=0, train_data = None, val_data = None, test_data = None):
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.batch_size = None
        self.index = index
        self.dataset = dataset
        self.dataset_to_eval = dataset.replace('Dataset_multy', 'Result_multy') if 'Dataset_multy' in dataset \
            else dataset.replace('Dataset_binary', 'Result_binary')

        if not os.path.exists(self.dataset_to_eval):
            os.makedirs(self.dataset_to_eval)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.model_paths = [f'{self.dataset_to_eval}/{y}/{y}.pt' for y in os.listdir(self.dataset_to_eval) if
                       '.csv' not in y and '.ini' not in y]
        self.encModel_paths = [f'{self.dataset_to_eval}/Enc{z}/' for z in os.listdir(self.dataset_to_eval) if
                          '.csv' not in z and '.ini' not in z]

        for enc_p  in self.encModel_paths:
            if os.path.exists(enc_p):
                shutil.rmtree(enc_p,ignore_errors=True)

        self.db_name = self.dataset_to_eval.split('/')[-1]
        self.context = self.generate_context()

        self.overall_res = os.path.join(self.dataset_to_eval, "overall_results.csv")

        # Intestazioni del CSV
        with open(self.overall_res, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Model", "Accuracy", "Precision", "Recall", "F1Score", "ConfusionMatrix", "Classification Report",
                 "Average_Loss"])

        self.num_classes = num_classes

        for model_path, encModel_path in tqdm(zip(self.model_paths, self.encModel_paths), desc=f'Enc Test ({dataset.split("/")[-1]})', position=index, colour='red', leave=False, total=len(self.model_paths)):
            filename = os.path.basename(model_path)
            match = re.search(r"model_h(\d+)_h2(\d+)_lr([0-9.e-]+)_bs(\d+)", filename)
            if not match:
                raise ValueError(f"Nome file non valido o non compatibile: {filename}")

            batch_size = int(match.group(4))

            self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

            os.makedirs(encModel_path, exist_ok=True)
            self.index = self.index + 1
            self.enc_test(model_path = model_path, output_path=f"{encModel_path}", index=self.index)


    def generate_context(self):
        bits_scale = 26

        # Create TenSEAL context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,  # Aumenta la capacità
            coeff_mod_bit_sizes=[60, bits_scale, bits_scale,
                                 bits_scale, bits_scale, bits_scale,
                                 bits_scale, bits_scale, bits_scale, bits_scale, 60]
        )

        # set the scale
        context.global_scale = pow(2, bits_scale)

        # galois keys are required to do ciphertext rotations
        context.generate_galois_keys()

        return context

    def load_model_for_encryption(self,model_path):

        # Estrai parametri dal nome del file
        filename = os.path.basename(model_path)
        match = re.search(r"model_h(\d+)_h2(\d+)_lr([0-9.e-]+)_bs(\d+)", filename)
        if not match:
            raise ValueError(f"Nome file non valido o non compatibile: {filename}")

        hidden = int(match.group(1))
        hidden2 = int(match.group(2))
        batch = int(match.group(4))

        # Carica lo state_dict
        state_dict = torch.load(model_path, map_location="cpu")

        # Deduce input_features dalla shape dei pesi
        fc1_weight_shape = state_dict['fc1.weight'].shape
        input_features = fc1_weight_shape[1]

        # print(hidden,hidden2,input_features)

        if input_features == 1600:
            image_size_general = 64
        elif input_features == 324:
            image_size_general = 32
        else:
            raise ValueError(f"Input features non riconosciuti: {input_features}")

        self.batch_size = batch

        # Istanzia il modello
        model = ConvNet(image_size_general_local=image_size_general,
                        hidden=hidden, hidden2=hidden2, num_classes=self.num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        return model, image_size_general

    def enc_test(self,model_path, output_path, index):

        model_name = (model_path.split("/")[-1]).replace(".pt", "")
        model, image_size_general_local = self.load_model_for_encryption(model_path)

        kernel_shape = model.conv1.kernel_size
        stride = model.conv1.stride[0]

        criterion = torch.nn.CrossEntropyLoss()
        enc_model = EncConvNet(model)

        test_loss = 0.0
        all_preds = []
        all_targets = []

        for data, target in tqdm(self.test_loader, desc=f'Test Enc ConvNet ({model_name.split("/")[-1]})', total=len(self.test_loader), 
                                 colour='yellow', leave=False, position=index):
            data = data.cpu()
            target = target.cpu()

            for i in range(len(data)):  # non assumere batch pieno
                single_data = data[i, 0]  # (1,H,W) -> (H,W)
                single_target = target[i]
                H = W = image_size_general_local

                x_enc, windows_nb = ts.im2col_encoding(
                    self.context,
                    single_data.view(H, W).tolist(),
                    kernel_shape[0],
                    kernel_shape[1],
                    stride
                )

                # Inference + Decryption
                enc_output = enc_model(x_enc, windows_nb)
                output = enc_output.decrypt()
                output = torch.tensor(output).view(1, -1)

                # Loss
                loss = criterion(output, single_target.unsqueeze(0))
                test_loss += loss.item()

                # Prediction
                _, pred = torch.max(output, 1)
                all_preds.append(pred.item())
                all_targets.append(single_target.item())

        # Metrics
        avg_loss = test_loss / len(all_targets)
        test_cnn(num_classes=self.num_classes,
                 all_preds=all_preds,
                 all_targets=all_targets,
                 tmp=None,
                 history=None,
                 out_put=f"{output_path}Result_Enc{model_name}.txt",
                 pdf_path = f"{output_path}Enc{model_name}_confusion_matrix.pdf",
                model_save_path = os.path.join(output_path, f"Enc{model_name}.pt"),
                model = model,
                overall_res = {'path': self.overall_res, 'model_name' : model_name, 'avg_loss':avg_loss}
                )

def test_cnn(num_classes, all_targets, all_preds, tmp, history, out_put,pdf_path, model_save_path, model, overall_res):

    disaply_label = ['normal', 'mild', 'severe'] if num_classes == 3 else ['0', '1']

    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=disaply_label,
                                   labels=[0, 1, 2] if num_classes == 3 else [0, 1])

    prec_w = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    prec_ma = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    prec_mi = precision_score(all_targets, all_preds, average='micro', zero_division=0)

    rec_w = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec_ma = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    rec_mi = recall_score(all_targets, all_preds, average='micro', zero_division=0)

    f1_w = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_ma = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_mi = f1_score(all_targets, all_preds, average='micro', zero_division=0)

    if num_classes == 2:
        prec_bi = precision_score(all_targets, all_preds, average='binary', zero_division=0)
        rec_bi = recall_score(all_targets, all_preds, average='binary', zero_division=0)
        f1_bi = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2] if num_classes == 3 else [0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()

    # Salva report txt
    with open(out_put, 'w') as f:
        f.write(f'Grid_Search Params\n')
        json.dump(tmp, f, indent=4)
        f.write(f"\n\n{'-' * 40}\n")
        f.write(f"Test Evaluation Report\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"{'-' * 40}\n")
        f.write(f"Weighted\n")
        f.write(f"Precision Weighted: {prec_w:.4f}\n")
        f.write(f"Recall Weighted: {rec_w:.4f}\n")
        f.write(f"F1 Score Weighted: {f1_w:.4f}\n\n")

        f.write(f"{'-' * 40}\n")
        f.write(f"Macro\n")
        f.write(f"Precision Macro: {prec_ma:.4f}\n")
        f.write(f"Recall Macro: {rec_ma:.4f}\n")
        f.write(f"F1 Score Macro: {f1_ma:.4f}\n\n")

        f.write(f"{'-' * 40}\n")
        f.write(f"Micro\n")
        f.write(f"Precision Micro: {prec_mi:.4f}\n")
        f.write(f"Recall Micro: {rec_mi:.4f}\n")
        f.write(f"F1 Score Micro: {f1_mi:.4f}\n\n")

        if num_classes == 2:
            f.write(f"{'-' * 40}\n")
            f.write(f"Binary\n")
            f.write(f"Precision Binary: {prec_bi:.4f}\n")
            f.write(f"Recall Binary: {rec_bi:.4f}\n")
            f.write(f"F1 Score Binary: {f1_bi:.4f}\n\n")
            f.write(f"{'-' * 40}\n")

        f.write(f"{'-' * 40}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")

        f.write(f"{'-' * 40}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

        f.write(f"\n{'-' * 40}\n")
        f.write("\n Report Epochs\n")

        if history:
            for h in history:
                f.write(f"{h}\n")

        if model_save_path and model:
            torch.save(model.state_dict(), model_save_path)

        if overall_res:
            cm_str = "; ".join(["[" + " ".join(map(str, row)) + "]" for row in cm])
            overall_res_path = overall_res['path']
            with open(overall_res_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                writer.writerow([overall_res['model_name'],  # "Model",
                                 f"{acc:.4f}",  # "Accuracy",
                                 f"{prec_w:.4f}",  # "Precision",
                                 f"{rec_w:.4f}",  # "Recall",
                                 f"{f1_w:.4f}",  # "F1Score",
                                 cm_str,  # "ConfusionMatrix",
                                 report,  # "Classification Report"
                                 f"{overall_res['avg_loss']:.4f}"  # Avg_loss
                                 ])