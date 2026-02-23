import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_npz_files(directory):
    npz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return npz_files

def save_data_splits(data_directories, base_path, seed, split_data, max_samples_per_dataset=None):
    train_paths = []
    val_paths = []
    test_paths = []

    for directory in data_directories:
        npz_file_paths = get_npz_files(directory)
        
        if max_samples_per_dataset:
            npz_file_paths = npz_file_paths[:max_samples_per_dataset]
        
        if split_data:
            train, test = train_test_split(npz_file_paths, test_size=0.4, random_state=seed)
            val, test = train_test_split(test, test_size=0.5, random_state=seed)
            
            train_paths.extend(train)
            val_paths.extend(val)
            test_paths.extend(test)
        else:
            train_paths.extend(npz_file_paths)

    if split_data:
        train_path = os.path.join(base_path, 'train_paths.txt')
        val_path = os.path.join(base_path, 'val_paths.txt')
        test_path = os.path.join(base_path, 'test_paths.txt')

        np.savetxt(train_path, train_paths, fmt='%s')
        np.savetxt(val_path, val_paths, fmt='%s')
        np.savetxt(test_path, test_paths, fmt='%s')

        print(f"Number of training samples: {len(train_paths)}")
        print(f"Number of validation samples: {len(val_paths)}")
        print(f"Number of testing samples: {len(test_paths)}")
    else:
        all_data_path = os.path.join(base_path, 'all_data_paths.txt')
        np.savetxt(all_data_path, train_paths, fmt='%s')
        print(f"Number of total samples: {len(train_paths)}")

    return train_paths, val_paths, test_paths

def main(data_directories, base_path, seed, split_data, max_samples_per_dataset=None):
    save_data_splits(data_directories, base_path, seed, split_data, max_samples_per_dataset)

if __name__ == '__main__':
    data_directories = [
    "/path/to/your/npz/files",
    ]

    base_path = "/path/to/save/split/data"
    os.makedirs(base_path, exist_ok=True)

    split_data = False
    max_samples_per_dataset = 10000

    main(data_directories, base_path, seed=42, split_data=split_data, max_samples_per_dataset=max_samples_per_dataset)
