'''
1. In the save_data_splits function, the decision of whether to perform data splitting is made based on the split_data parameter.
2. In the main function, the choice of whether to print the label distribution of each dataset or the label distribution of the entire dataset is determined by the split_data parameter.
3. split_data is a global parameter used to control whether to perform dataset splitting.
'''
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

def load_paths_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

def load_npz_labels(npz_files):
    all_labels = []
    for file in npz_files:
        data = np.load(file)
        labels = data['y']
        all_labels.extend(labels)
    return np.array(all_labels)

def count_labels(labels):
    label_counts = np.bincount(labels, minlength=5)  # Assuming there are 5 labels (0, 1, 2, 3, 4)
    return label_counts

def print_label_distribution(name, labels):
    total = labels.sum()
    print(f"{name} set label distribution:")
    for i, count in enumerate(labels):
        print(f"Label {i}: {count} ({count / total:.2%})")
    print()

def main(data_directories, base_path, seed, split_data, max_samples_per_dataset=None):
    # Step 1: Split the data
    train_paths, val_paths, test_paths = save_data_splits(data_directories, base_path, seed, split_data, max_samples_per_dataset)
    
    if split_data:
        # Step 2: Load labels
        train_labels = load_npz_labels(train_paths)
        val_labels = load_npz_labels(val_paths)
        test_labels = load_npz_labels(test_paths)

        # Step 3: Count labels
        train_label_counts = count_labels(train_labels)
        val_label_counts = count_labels(val_labels)
        test_label_counts = count_labels(test_labels)

        # Step 4: Print label distribution
        print_label_distribution("Train", train_label_counts)
        print_label_distribution("Validation", val_label_counts)
        print_label_distribution("Test", test_label_counts)
    else:
        all_labels = load_npz_labels(train_paths)
        all_label_counts = count_labels(all_labels)
        print_label_distribution("All data", all_label_counts)

if __name__ == '__main__':
    data_directories = [
    "path/to/dataset1",
    ]

    base_path = 'path/to/save/splits'
    os.makedirs(base_path, exist_ok=True)

    split_data = True
    max_samples_per_dataset = 5000

    main(data_directories, base_path, seed=42, split_data=split_data, max_samples_per_dataset=max_samples_per_dataset)
