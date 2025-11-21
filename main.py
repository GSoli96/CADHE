import os
import shutil

import pandas as pd
from torch.distributions import identity_transform

from CNN import train
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import itertools

from CNN.train import train_test_EncConvNet

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from tqdm import tqdm
import pandas as pd
import os

from CNN import train
from CNN.train import train_test_EncConvNet
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def run_grid_search(data_dir, param_grid, image_size_general):
    dataset_name = data_dir
    transform = transforms.Compose([
        transforms.Resize((image_size_general, image_size_general)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)

    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

    db_name = dataset_name.split('/')[-1]
    res_db = f"Result_binary/{db_name}/" if 'binary' in db_name else f"Result_multy/{db_name}/"
    os.makedirs(res_db, exist_ok=True)

    results = []
    combinations = list(itertools.product(
        param_grid['hidden'], param_grid['hidden2'],
        param_grid['lr'], param_grid['batch_size']
    ))

    def train_one_combo(hidden, hidden2, lr, batch_size, idx):
        grid_path = os.path.join(res_db, f'model_h{hidden}_h2{hidden2}_lr{lr}_bs{batch_size}')
        if os.path.exists(grid_path):
            return None

        os.makedirs(grid_path, exist_ok=True)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        istance_train_test = train.train_test_ConvNet(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            hidden=hidden,
            hidden2=hidden2,
            lr=lr,
            batch_size=batch_size,
            image_size_general=image_size_general,
            grid_path=grid_path,
            num_classes=num_classes,
            epochs=50,
            position=idx
        )

        return istance_train_test.test_ConvNet()

    # Parallelizza 2 combinazioni alla volta
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, (hidden, hidden2, lr, batch_size) in enumerate(combinations, start=0):
            futures.append(executor.submit(train_one_combo, hidden, hidden2, lr, batch_size, idx))

        for f in tqdm(as_completed(futures), total=len(futures), desc="GridSearch (Parallel 2)", colour="cyan"):
            res = f.result()
            if res is not None:
                results.append(res)

    # Ordina e salva risultati
    results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(res_db, f"ParamGrid_{db_name}.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Avvia test cifrato
    train_test_EncConvNet(dataset=data_dir, num_classes=num_classes, index=0,
                          train_data=train_data, val_data=val_data, test_data=test_data)

def main():
    datasets = [
        # "Dataset_binary/dataset_slices_sorted",
        # "Dataset_binary/dataset_slices_sorted_T1",
        # "Dataset_binary/dataset_slices_sorted_T2",
        # "Dataset_binary/dataset_slices_sorted_Flair",
        # "Dataset_binary/dataset_slices_sorted_augmented",
        # "Dataset_binary/dataset_slices_sorted_T1_augmented",
        # "Dataset_binary/dataset_slices_sorted_T2_augmented",
        # "Dataset_binary/dataset_slices_sorted_Flair_augmented",
        "Dataset_multy/dataset_slices_sorted",
        "Dataset_multy/dataset_slices_sorted_T1",
        "Dataset_multy/dataset_slices_sorted_T2",
        "Dataset_multy/dataset_slices_sorted_Flair",
        "Dataset_multy/dataset_slices_sorted_augmented",
        "Dataset_multy/dataset_slices_sorted_T1_augmented",
        "Dataset_multy/dataset_slices_sorted_T2_augmented",
        "Dataset_multy/dataset_slices_sorted_Flair_augmented",
    ]

    # Griglia dei parametri estesa
    param_grid = {
        'hidden': [32],
        'hidden2': [16],
        'lr': [1e-2, 1e-3, 1e-4],
        'batch_size': [32, 64]
    }
    image_size_general = 32

    # Imposta quanti processi paralleli vuoi (es. 4)
    for dataset in datasets:
        run_grid_search(data_dir=dataset,param_grid=param_grid, image_size_general=image_size_general)

if __name__ == "__main__":
    main()
