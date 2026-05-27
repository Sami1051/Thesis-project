import os
import glob
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torch.distributed.algorithms.join import Join as mjoin
import torchmetrics

from torch.utils.tensorboard import SummaryWriter

# Import the TFRecord reade
# pip install tfrecord
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
import numpy as np

from pytorch_metric_learning import losses

#exit(0)

'''
def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    g=torch.cuda.CUDAGraph()
    del g
    dist.destroy_process_group()
    '''



# Implementation: Label adjustment using "single-centroid" method


def load_soft_labels_from_features(feature_path, oil_label=0, clutter_label=3):
    data = np.load(feature_path)

    features = data["features"]
    labels = data["labels"]
    indexes = data["indexes"]

    # Filter only oil and clutter
    mask = (labels == oil_label) | (labels == clutter_label)

    features = features[mask]
    labels = labels[mask]
    indexes = indexes[mask]

    oil_features = features[labels == oil_label]
    clutter_features = features[labels == clutter_label]

    # Compute centers
    oil_center = np.mean(oil_features, axis=0)
    clutter_center = np.mean(clutter_features, axis=0)

    soft_labels = {}

    for f, idx in zip(features, indexes):
        d_oil = np.linalg.norm(f - oil_center)
        d_clutter = np.linalg.norm(f - clutter_center)

        total = d_oil + d_clutter + 1e-8

        w_oil = d_clutter / total
        w_clutter = d_oil / total

        soft = np.zeros(4)
        soft[oil_label] = w_oil
        soft[clutter_label] = w_clutter

        soft_labels[int(idx)] = soft

    print(f"Loaded soft labels for {len(soft_labels)} samples")
    print("Used 1 oil center and 1 clutter center")

    return soft_labels



# Implementation: Label adjustment using "multi-centers" method 

'''
def load_soft_labels_from_features(feature_path, oil_label=0, clutter_label=3, n_centers=3):  # Added n_centers=3 for multi-centers
    from sklearn.cluster import KMeans
    
    data = np.load(feature_path)

    features = data["features"]
    labels = data["labels"]
    indexes = data["indexes"]

    # Filter only oil and clutter
    mask = (labels == oil_label) | (labels == clutter_label)

    features = features[mask]
    labels = labels[mask]
    indexes = indexes[mask]

    oil_features = features[labels == oil_label]
    clutter_features = features[labels == clutter_label]

    # Compute centers
    #oil_center = np.mean(oil_features, axis=0)             # For single center
    #clutter_center = np.mean(clutter_features, axis=0)     # For single center

    n_oil_centers = min(n_centers, len(oil_features))
    n_clutter_centers = min(n_centers, len(clutter_features))

    oil_kmeans = KMeans(n_clusters=n_oil_centers, random_state=42, n_init=10)
    clutter_kmeans = KMeans(n_clusters=n_clutter_centers, random_state=42, n_init=10)

    oil_kmeans.fit(oil_features)
    clutter_kmeans.fit(clutter_features)

    oil_centers = oil_kmeans.cluster_centers_
    clutter_centers = clutter_kmeans.cluster_centers_


    soft_labels = {}

    for f, idx in zip(features, indexes):
        #d_oil = np.linalg.norm(f - oil_center)          # For single center
        #d_clutter = np.linalg.norm(f - clutter_center)  # For single center

        d_oil = np.min(np.linalg.norm(oil_centers - f, axis=1))
        d_clutter = np.min(np.linalg.norm(clutter_centers - f, axis=1))


        total = d_oil + d_clutter + 1e-8

        w_oil = d_clutter / total
        w_clutter = d_oil / total

        soft = np.zeros(4)
        soft[oil_label] = w_oil
        soft[clutter_label] = w_clutter

        soft_labels[int(idx)] = soft

    print(f"Loaded soft labels for {len(soft_labels)} samples")
    print(f"Used {n_oil_centers} oil centers and {n_clutter_centers} clutter centers")

    return soft_labels
'''

# Implementation: Label adjustment using "nearest neighbors" method (knn)

'''
def load_soft_labels_from_features(feature_path, oil_label=0, clutter_label=3, n_centers=3, k_neighbors=5):  # Added k_neighbors=5 for nearest-neighbor soft labels
    from sklearn.neighbors import NearestNeighbors
    
    data = np.load(feature_path)

    features = data["features"]
    labels = data["labels"]
    indexes = data["indexes"]

    # Filter only oil and clutter
    mask = (labels == oil_label) | (labels == clutter_label)

    features = features[mask]
    labels = labels[mask]
    indexes = indexes[mask]

    oil_features = features[labels == oil_label]
    clutter_features = features[labels == clutter_label]

    oil_indexes = indexes[labels == oil_label]
    clutter_indexes = indexes[labels == clutter_label]

    # Compute centers
    #oil_center = np.mean(oil_features, axis=0)             # For single center
    #clutter_center = np.mean(clutter_features, axis=0)     # For single center

    # Nearest-neighbor setup
    k_oil = min(k_neighbors + 1, len(oil_features))
    k_clutter = min(k_neighbors + 1, len(clutter_features))

    oil_nn = NearestNeighbors(n_neighbors=k_oil, metric="euclidean", algorithm="brute", n_jobs=1)
    clutter_nn = NearestNeighbors(n_neighbors=k_clutter, metric="euclidean", algorithm="brute", n_jobs=1)

    oil_nn.fit(oil_features)
    clutter_nn.fit(clutter_features)

    oil_distances, oil_neighbor_indices = oil_nn.kneighbors(features)
    clutter_distances, clutter_neighbor_indices = clutter_nn.kneighbors(features)

    soft_labels = {}

    for row_i, (f, idx) in enumerate(zip(features, indexes)):
        #d_oil = np.linalg.norm(f - oil_center)          # For single center
        #d_clutter = np.linalg.norm(f - clutter_center)  # For single center

        oil_dists = oil_distances[row_i]
        oil_neigh_idx = oil_neighbor_indices[row_i]
        oil_neigh_indexes = oil_indexes[oil_neigh_idx]

        clutter_dists = clutter_distances[row_i]
        clutter_neigh_idx = clutter_neighbor_indices[row_i]
        clutter_neigh_indexes = clutter_indexes[clutter_neigh_idx]

        # Remove self-neighbor if the sample itself is included
        oil_dists = oil_dists[oil_neigh_indexes != idx]
        clutter_dists = clutter_dists[clutter_neigh_indexes != idx]

        # Use k nearest neighbours after removing self
        oil_dists = oil_dists[:k_neighbors]
        clutter_dists = clutter_dists[:k_neighbors]

        d_oil = np.mean(oil_dists)
        d_clutter = np.mean(clutter_dists)

        total = d_oil + d_clutter + 1e-8

        w_oil = d_clutter / total
        w_clutter = d_oil / total

        soft = np.zeros(4)
        soft[oil_label] = w_oil
        soft[clutter_label] = w_clutter

        soft_labels[int(idx)] = soft

    print(f"Loaded soft labels for {len(soft_labels)} samples")
    print(f"Used {k_neighbors} nearest oil neighbours and {k_neighbors} nearest clutter neighbours")

    return soft_labels

'''


def physically_realizable_oil_drum_augmentation(image):
    augmentation = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
    ])

    return augmentation(image)


def non_physically_realizable_oil_drum_augmentation(image):
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomChoice([
            transforms.RandomRotation(
                degrees=(30, 45),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            ),
            transforms.RandomRotation(
                degrees=(-45, -30),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0
            ),
        ]),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.75, 1.25),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
    ])

    return augmentation(image)

'''
def decode_image(features):
    """
    Callback to decode raw bytes into a PyTorch tensor.
    Customize this based on how your TFRecords were written.
    """
    import io
    from PIL import Image
    
    # Get raw bytes and label
    # NOTE: Check your TFRecord keys (e.g., 'image_raw', 'image', 'label')
    image_bytes = features["image/encoded"]
    label = features["image/class/label"]
    index= features["image/index"]

    # Decode image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)

    ## Label mapping and one-hot encoding here.
    map_dic_nm={53:'detector_clutter', 26:'oil_drum', 49:'other_man_made_object', 51:'mine_size_rock'}
    map_dic={52:3, 25:0, 48:1, 50:2}


    #print('lable shape ' ,np.shape(label), ' ', label[0])
    mapped_label= map_dic[label[0]] if label[0] in map_dic else 3
    
    # Return dictionary or tuple
    return image, torch.tensor(mapped_label, dtype=torch.long), torch.tensor(index)
'''


# For augmentation experiment

def decode_image(features, augment_oil_drum=False):
    """
    Callback to decode raw bytes into a PyTorch tensor.
    Customize this based on how your TFRecords were written.
    """
    import io
    from PIL import Image
    
    # Get raw bytes and label
    # NOTE: Check your TFRecord keys (e.g., 'image_raw', 'image', 'label')
    image_bytes = features["image/encoded"]
    label = features["image/class/label"]
    index= features["image/index"]

    # Decode image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    ## Label mapping and one-hot encoding here.
    map_dic_nm={53:'detector_clutter', 26:'oil_drum', 49:'other_man_made_object', 51:'mine_size_rock'}
    map_dic={52:3, 25:0, 48:1, 50:2}


    #print('lable shape ' ,np.shape(label), ' ', label[0])
    mapped_label= map_dic[label[0]] if label[0] in map_dic else 3

    if augment_oil_drum and mapped_label == 0:
        image = physically_realizable_oil_drum_augmentation(image)
        #image = non_physically_realizable_oil_drum_augmentation(image)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    
    # Return dictionary or tuple
    return image, torch.tensor(mapped_label, dtype=torch.long), torch.tensor(index)


def get_dataloader(tfrecord_pattern, batch_size=1, loader_name="data", augment_oil_drum=False):  # Added augment_oil_drum=False for augmentation experiment
    """
    Creates a DataLoader that reads only the shard of files belonging to this rank.
    """
    # 1. Find all TFRecord files
    #all_files = sorted(glob.glob(tfrecord_pattern))
    all_files = sorted(
    glob.glob(tfrecord_pattern),
    key=lambda x: int(x.split('_i')[-1].split('.tfrecord')[0])
)

    if len(all_files) == 0:
        print(f"[{loader_name}] No files found for pattern: {tfrecord_pattern}")

    # 2. Shard the files: Each GPU gets a unique subset of files
    # This replaces DistributedSampler for IterableDatasets
    #my_files = all_files[rank::world_size]
    
    
    my_files=all_files

    my_splits={} 
    my_ind_files=[] 
    for i, my_file in enumerate(my_files):
        my_splits[i]=1.0
        my_ind_file_cur=my_file.replace('tfrecord', 'tfindex')
        my_ind_files=my_ind_files+[my_ind_file_cur]

    if not my_files:
        print(f"Warning: There is  no files to process!")
    else:
        print(f"[{loader_name}] Found {len(all_files)} TFRecord files")
        print(f"[{loader_name}] First file: {all_files[0]}")
        print(f"[{loader_name}] Last file: {all_files[-1]}")
        print(f"[{loader_name}] Using {len(my_files)} TFRecord files and {len(my_ind_files)} index files")



    dataset = MultiTFRecordDataset(
        data_pattern=my_files,
        index_pattern=my_ind_files,
        splits=my_splits,
        #data_pattern=tfrecord_pattern,
        #index_pattern=index_pattern,  # Set None if you don't have index files
        #splits={'i0': 1.0, 'i1':1.0},# 'i24':0.5},
        description={"image/encoded": "byte", "image/class/label": "int", "image/index": "int"},  # Describe data types
        #transform=decode_image,  # decoding function
        transform=lambda features: decode_image(features, augment_oil_drum=augment_oil_drum),  # decoding function
        shuffle_queue_size=1024,  # Shuffle buffer size for randomness
        infinite=False
    )

    # 4. Create DataLoader
    # num_workers > 0 ensures data loading happens in parallel processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2
    )

    return loader


def get_class_distribution(loader, name):
    counts = {}

    for images, labels, _ in loader:
        for l in labels.numpy():
            counts[l] = counts.get(l, 0) + 1

    total = sum(counts.values())

    print(f"\n{name} distribution:")
    for k in sorted(counts.keys()):
        print(f"Class {k}: {counts[k]/total:.2%}")


def extract_feature_vectors(model, loader, device, split_name, save_dir):
    model.eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    all_features = []
    all_labels = []
    all_indexes = []
    all_predicted_labels = []
    all_probs = []

    print(f"Start feature extraction for {split_name}")

    with torch.no_grad():
        for i, (images, labels, indexes) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feature_maps = feature_extractor(images)
            features = torch.flatten(feature_maps, 1)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1)

            all_features.append(features.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))
            all_indexes.append(indexes.detach().cpu().numpy().reshape(-1))
            all_predicted_labels.append(predicted_labels.detach().cpu().numpy().reshape(-1))
            all_probs.append(probs.detach().cpu().numpy())

            if i % 100 == 0:
                print(f"[{split_name}] Feature extraction step [{i}]")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_indexes = np.concatenate(all_indexes, axis=0)
    all_predicted_labels = np.concatenate(all_predicted_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    save_path = os.path.join(save_dir, f"{split_name}_features.npz")
    np.savez(
        save_path,
        features=all_features,
        labels=all_labels,
        indexes=all_indexes,
        predicted_labels=all_predicted_labels,
        probs=all_probs
    )

    print(f"[{split_name}] Saved feature vectors to: {save_path}")
    print(f"[{split_name}] Features shape: {all_features.shape}")

    return save_path


def analyze_feature_space(train_feature_file, val_feature_file, save_dir, oil_drum_label=0, clutter_label=3, top_k=5):
    print("Start feature-space nearest-neighbour analysis")

    train_data = np.load(train_feature_file)
    val_data = np.load(val_feature_file)

    train_features = train_data["features"]
    train_labels = train_data["labels"]
    train_indexes = train_data["indexes"]
    train_predicted_labels = train_data["predicted_labels"]
    train_probs = train_data["probs"]

    val_features = val_data["features"]
    val_labels = val_data["labels"]
    val_indexes = val_data["indexes"]
    val_predicted_labels = val_data["predicted_labels"]
    val_probs = val_data["probs"]

    misclassified_oil_mask = (val_labels == oil_drum_label) & (val_predicted_labels == clutter_label)
    clutter_train_mask = (train_labels == clutter_label)

    query_features = val_features[misclassified_oil_mask]
    query_indexes = val_indexes[misclassified_oil_mask]
    query_labels = val_labels[misclassified_oil_mask]
    query_predicted_labels = val_predicted_labels[misclassified_oil_mask]
    query_probs = val_probs[misclassified_oil_mask]

    reference_features = train_features[clutter_train_mask]
    reference_indexes = train_indexes[clutter_train_mask]
    reference_labels = train_labels[clutter_train_mask]
    reference_predicted_labels = train_predicted_labels[clutter_train_mask]
    reference_probs = train_probs[clutter_train_mask]

    summary_path = os.path.join(save_dir, "misclassified_oil_drum_queries.csv")
    neighbour_path = os.path.join(save_dir, f"nearest_clutter_top_{top_k}.csv")

    with open(summary_path, mode="w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "query_index",
            "query_true_label",
            "query_predicted_label",
            "query_prob_class_0",
            "query_prob_class_1",
            "query_prob_class_2",
            "query_prob_class_3"
        ])

        for q_idx, q_true, q_pred, q_prob in zip(
            query_indexes, query_labels, query_predicted_labels, query_probs
        ):
            summary_writer.writerow([
                int(q_idx),
                int(q_true),
                int(q_pred),
                float(q_prob[0]),
                float(q_prob[1]),
                float(q_prob[2]),
                float(q_prob[3])
            ])

    with open(neighbour_path, mode="w", newline="") as neighbour_file:
        neighbour_writer = csv.writer(neighbour_file)
        neighbour_writer.writerow([
            "query_index",
            "query_true_label",
            "query_predicted_label",
            "query_prob_class_0",
            "query_prob_class_1",
            "query_prob_class_2",
            "query_prob_class_3",
            "neighbor_rank",
            "neighbor_split",
            "neighbor_index",
            "neighbor_true_label",
            "neighbor_predicted_label",
            "neighbor_prob_class_0",
            "neighbor_prob_class_1",
            "neighbor_prob_class_2",
            "neighbor_prob_class_3",
            "euclidean_distance"
        ])

        if len(query_features) == 0:
            print("No validation oil-drum samples misclassified as clutter were found.")
        elif len(reference_features) == 0:
            print("No train clutter samples were found for nearest-neighbour search.")
        else:
            for q_i in range(len(query_features)):
                q_feature = query_features[q_i]
                distances = np.linalg.norm(reference_features - q_feature, axis=1)
                nearest_indices = np.argsort(distances)[:top_k]

                for rank, ref_idx in enumerate(nearest_indices, start=1):
                    q_prob = query_probs[q_i]
                    ref_prob = reference_probs[ref_idx]

                    neighbour_writer.writerow([
                        int(query_indexes[q_i]),
                        int(query_labels[q_i]),
                        int(query_predicted_labels[q_i]),
                        float(q_prob[0]),
                        float(q_prob[1]),
                        float(q_prob[2]),
                        float(q_prob[3]),
                        rank,
                        "train",
                        int(reference_indexes[ref_idx]),
                        int(reference_labels[ref_idx]),
                        int(reference_predicted_labels[ref_idx]),
                        float(ref_prob[0]),
                        float(ref_prob[1]),
                        float(ref_prob[2]),
                        float(ref_prob[3]),
                        float(distances[ref_idx])
                    ])

                if q_i % 10 == 0:
                    print(f"Processed nearest neighbours for query [{q_i}]")

    print(f"Saved misclassified oil-drum query summary to: {summary_path}")
    print(f"Saved nearest clutter neighbour analysis to: {neighbour_path}")


def run_feature_space_analysis(model, train_loader, val_loader, device, save_dir, oil_drum_label=0, clutter_label=3, top_k=5):
    os.makedirs(save_dir, exist_ok=True)

    train_feature_file = extract_feature_vectors(
        model=model,
        loader=train_loader,
        device=device,
        split_name="train",
        save_dir=save_dir
    )

    val_feature_file = extract_feature_vectors(
        model=model,
        loader=val_loader,
        device=device,
        split_name="validation",
        save_dir=save_dir
    )

    analyze_feature_space(
        train_feature_file=train_feature_file,
        val_feature_file=val_feature_file,
        save_dir=save_dir,
        oil_drum_label=oil_drum_label,
        clutter_label=clutter_label,
        top_k=5
    )



class FeatureDomainClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def create_smote_feature_datasets(train_feature_file, val_feature_file, oil_label=0, num_classes=4, k_neighbors=5, random_state=42):
    from sklearn.neighbors import NearestNeighbors
    from torch.utils.data import TensorDataset

    rng = np.random.default_rng(random_state)

    train_data = np.load(train_feature_file)
    val_data = np.load(val_feature_file)

    train_features = train_data["features"].astype(np.float32)
    train_labels = train_data["labels"].astype(np.int64)

    val_features = val_data["features"].astype(np.float32)
    val_labels = val_data["labels"].astype(np.int64)

    oil_features = train_features[train_labels == oil_label]

    class_counts = np.bincount(train_labels, minlength=num_classes)
    #target_oil_count = int(np.max(class_counts))  # oil-drum features matched to the biggest class (clutter)
    target_oil_count = len(oil_features) * 100     # 100x oil drum
    n_synthetic = max(0, target_oil_count - len(oil_features))

    if len(oil_features) < 2:
        synthetic_features = np.empty((0, train_features.shape[1]), dtype=np.float32)
        synthetic_labels = np.empty((0,), dtype=np.int64)
    else:
        k = min(k_neighbors + 1, len(oil_features))
        nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute", n_jobs=1)
        nn_model.fit(oil_features)
        _, neighbor_indices = nn_model.kneighbors(oil_features)

        synthetic_features = []

        for _ in range(n_synthetic):
            base_i = rng.integers(0, len(oil_features))
            possible_neighbors = neighbor_indices[base_i][1:]

            if len(possible_neighbors) == 0:
                neighbor_i = base_i
            else:
                neighbor_i = rng.choice(possible_neighbors)

            lam = rng.random()  # random interpolation weight between 0 and 1 is chosen
            synthetic = oil_features[base_i] + lam * (oil_features[neighbor_i] - oil_features[base_i])
            synthetic_features.append(synthetic)

        synthetic_features = np.array(synthetic_features, dtype=np.float32)
        synthetic_labels = np.full((len(synthetic_features),), oil_label, dtype=np.int64)

    augmented_train_features = np.concatenate([train_features, synthetic_features], axis=0)
    augmented_train_labels = np.concatenate([train_labels, synthetic_labels], axis=0)

    permutation = rng.permutation(len(augmented_train_labels))
    augmented_train_features = augmented_train_features[permutation]
    augmented_train_labels = augmented_train_labels[permutation]

    train_dataset = TensorDataset(
        torch.tensor(augmented_train_features, dtype=torch.float32),
        torch.tensor(augmented_train_labels, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )

    print("Original train class distribution:", class_counts)
    print(f"Original oil-drum samples: {len(oil_features)}")
    print(f"Synthetic oil-drum samples generated: {len(synthetic_features)}")
    print("Augmented train class distribution:", np.bincount(augmented_train_labels, minlength=num_classes))

    return train_dataset, val_dataset, train_features.shape[1]


def train_smote_classifier(model, train_loader, optimizer, criterion, metrics, device):
    acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix = metrics

    model.train()
    running_loss = 0.0

    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(features)

        acc = acc_metric(outputs, labels)
        precision = precision_metric(outputs, labels)
        recall = recall_metric(outputs, labels)
        f1 = f1_metric(outputs, labels)
        confmat = conf_matrix(outputs, labels)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f" Step [{i}] Loss: {loss.item():.4f}")

    ave_running_loss = torch.tensor(running_loss / (i + 1)).to(device)

    accuracy = acc_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    confmat = conf_matrix.compute()

    print(f"Epoch finished. Avg ave Loss: {ave_running_loss.item():.4f}")
    print("Accuracy:", accuracy.item())
    print("Precision:", precision.item())
    print("Recall:", recall.item())
    print("F1 Score:", f1.item())
    print("Confusion Matrix", confmat.cpu().numpy())

    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    conf_matrix.reset()

    return ave_running_loss.item(), accuracy.item(), precision.item(), recall.item(), f1.item(), confmat.cpu().numpy()


def validate_smote_classifier(model, val_loader, criterion, metrics, device):
    acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix = metrics

    model.eval()
    running_loss = 0.0

    print("start validation")

    with torch.no_grad():
        for i, (features, labels) in enumerate(val_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(features)

            acc = acc_metric(outputs, labels)
            precision = precision_metric(outputs, labels)
            recall = recall_metric(outputs, labels)
            f1 = f1_metric(outputs, labels)
            confmat = conf_matrix(outputs, labels)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"\tStep [{i}] Loss: {loss.item():.4f}")

    ave_running_loss = torch.tensor(running_loss / (i + 1)).to(device)

    accuracy = acc_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    confmat = conf_matrix.compute()

    print(f"Epoch validation finished. Avg ave Loss: {ave_running_loss.item():.4f}")
    print("Accuracy (validation):", accuracy.item())
    print("Precision (validation):", precision.item())
    print("Recall (validation):", recall.item())
    print("F1 Score (validation):", f1.item())
    print("Confusion Matrix (validation)", confmat.cpu().numpy())

    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    conf_matrix.reset()

    return ave_running_loss.item(), accuracy.item(), precision.item(), recall.item(), f1.item(), confmat.cpu().numpy()


def run_feature_domain_smote_experiment(train_feature_file, val_feature_file, log_dir, summary_writer, device, epochs=25):
    import matplotlib.pyplot as plt

    train_dataset, val_dataset, input_dim = create_smote_feature_datasets(
        train_feature_file=train_feature_file,
        val_feature_file=val_feature_file,
        oil_label=0,
        num_classes=4,
        k_neighbors=5,
        random_state=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=20,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    model = FeatureDomainClassifier(input_dim=input_dim, num_classes=4)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)
    conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=4).to(device)

    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=4, average="macro").to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=4, average="macro").to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro").to(device)

    class_names = [str(i) for i in range(4)]

    best_vl_loss = 1000

    for epoch in range(epochs):
        print("Epoch", epoch)

        tr_loss, tr_acc, tr_precision, tr_recall, tr_f1, tr_cmat = train_smote_classifier(
            model, train_loader, optimizer, criterion,
            [acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix], device
        )

        summary_writer.add_scalar("Loss/train", tr_loss, global_step=epoch)
        summary_writer.add_scalar("Accuracy/train", tr_acc, global_step=epoch)
        summary_writer.add_scalar("Precision/train", tr_precision, global_step=epoch)
        summary_writer.add_scalar("Recall/train", tr_recall, global_step=epoch)
        summary_writer.add_scalar("F1/train", tr_f1, global_step=epoch)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(tr_cmat, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix - Train"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(tr_cmat.shape[0]):
            for j in range(tr_cmat.shape[1]):
                ax.text(j, i, int(tr_cmat[i, j]), ha="center", va="center")
        fig.tight_layout()
        summary_writer.add_figure("ConfusionMatrix/train", fig, global_step=epoch)
        plt.close(fig)

        vl_loss, vl_acc, vl_precision, vl_recall, vl_f1, vl_cmat = validate_smote_classifier(
            model, val_loader, criterion,
            [acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix], device
        )

        summary_writer.add_scalar("Loss/validation", vl_loss, global_step=epoch)
        summary_writer.add_scalar("Accuracy/validation", vl_acc, global_step=epoch)
        summary_writer.add_scalar("Precision/validation", vl_precision, global_step=epoch)
        summary_writer.add_scalar("Recall/validation", vl_recall, global_step=epoch)
        summary_writer.add_scalar("F1/validation", vl_f1, global_step=epoch)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(vl_cmat, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix - Validation"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(vl_cmat.shape[0]):
            for j in range(vl_cmat.shape[1]):
                ax.text(j, i, int(vl_cmat[i, j]), ha="center", va="center")
        fig.tight_layout()
        summary_writer.add_figure("ConfusionMatrix/validation", fig, global_step=epoch)
        plt.close(fig)

        if vl_loss < best_vl_loss:
            torch.save(model.state_dict(), os.path.join(log_dir, "feature_domain_smote_classifier.pth"))
            best_vl_loss = vl_loss

    print("Done")




def oversample_oil_drum_batch(images, labels, indexes, oil_label=0, factor=100):
    if factor <= 1:
        return images, labels, indexes

    oil_mask = labels == oil_label

    if not oil_mask.any():
        return images, labels, indexes

    oil_images = images[oil_mask].repeat_interleave(factor - 1, dim=0)
    oil_labels = labels[oil_mask].repeat_interleave(factor - 1, dim=0)
    oil_indexes = indexes[oil_mask].repeat_interleave(factor - 1, dim=0)

    images = torch.cat([images, oil_images], dim=0)
    labels = torch.cat([labels, oil_labels], dim=0)
    indexes = torch.cat([indexes, oil_indexes], dim=0)

    permutation = torch.randperm(labels.size(0))

    images = images[permutation]
    labels = labels[permutation]
    indexes = indexes[permutation]

    return images, labels, indexes





def train_fn(device, tfrecord_pattern_trn,tfrecord_pattern_val, epochs=20):
    print( ' TF patttern_trn ', tfrecord_pattern_trn, ' TF_pattern_val ', tfrecord_pattern_val)


    #summary_writer = SummaryWriter(log_dir="runs/exp1")

    base_dir = "runs/experiment_15_augmentation_physical_oversampled"
    os.makedirs(base_dir, exist_ok=True)

    # Finn eksisterende run-mapper
    existing_runs = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]

    # Ekstra robust: finn høyeste nummer (ikke bare len)
    run_numbers = [
        int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()
    ]

    next_run = max(run_numbers, default=0) + 1

    log_dir = os.path.join(base_dir, f"run_{next_run:03d}")
    os.makedirs(log_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=log_dir)

    print(f"Logging til: {log_dir}")

    oil_drum_label = 0
    oil_drum_tracking_dir = os.path.join(log_dir, "oil_drum_tracking")
    os.makedirs(oil_drum_tracking_dir, exist_ok=True)


    feature_analysis_dir = os.path.join(log_dir, "feature_analysis")
    os.makedirs(feature_analysis_dir, exist_ok=True)

    '''
    run_feature_domain_smote_experiment(
        train_feature_file="runs/experiment_04_feature_space_analysis/run_001/feature_analysis/train_features.npz",
        val_feature_file="runs/experiment_04_feature_space_analysis/run_001/feature_analysis/validation_features.npz",
        log_dir=log_dir,
        summary_writer=summary_writer,
        device=device,
        epochs=epochs
    )

    return
    '''

    
    # --- 1. Data Loading ---
    train_loader = get_dataloader(tfrecord_pattern_trn, batch_size=20, loader_name="train", augment_oil_drum=True)
    val_loader = get_dataloader(tfrecord_pattern_val, batch_size=20, loader_name="validation", augment_oil_drum=False)

    '''
    soft_label_dict = load_soft_labels_from_features(
        "runs/experiment_04_feature_space_analysis/run_001/feature_analysis/train_features.npz",
        n_centers=3
        #k_neighbors=5  
    )
    print(len(soft_label_dict))
    '''

    soft_label_dict = None
    print(soft_label_dict)

    #get_class_distribution(train_loader, "Train")
    #get_class_distribution(val_loader, "Validation")


    # --- 2. Model Setup ---
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_fea= model.fc.in_features
    model.fc=nn.Linear(num_fea, 4)

    model=model.to(device)

    retrain=False 
    PATH = 'logs/exp1.pth'
    if (retrain):
        print('Starting from a previous checkpoint...')
        model.load_state_dict(torch.load(PATH))


    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    #triplet_loss_func = losses.TripletMarginLoss()   # New 

    acc_metric=torchmetrics.Accuracy(task="multiclass",num_classes=4).to(device)
    conf_matrix=torchmetrics.ConfusionMatrix(task="multiclass", num_classes=4).to(device)

    # Nye metrics
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=4, average="macro").to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=4, average="macro").to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro").to(device)

    # Nytt: imports og klassenavn for confusion matrix plotting i TensorBoard
    import matplotlib.pyplot as plt
    class_names = [str(i) for i in range(4)]

    best_vl_loss=1000
    # --- 3. Training Loop ---
    for epoch in range(epochs):
        print('Epoch', epoch)

        #if epoch == 10:
            #print("Freezing feature extractor...")
            #for param in model.parameters():
                #param.requires_grad = False

            #for param in model.fc.parameters():
                #param.requires_grad = True

            #optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

        tr_loss, tr_acc, tr_precision, tr_recall, tr_f1, tr_cmat = train(
            model, train_loader, optimizer, criterion, #triplet_loss_func,
            [acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix], device,
            soft_label_dict
        )
        summary_writer.add_scalar('Loss/train', tr_loss, global_step=epoch)
        summary_writer.add_scalar('Accuracy/train', tr_acc, global_step=epoch)
        summary_writer.add_scalar('Precision/train', tr_precision, global_step=epoch)
        summary_writer.add_scalar('Recall/train', tr_recall, global_step=epoch)
        summary_writer.add_scalar('F1/train', tr_f1, global_step=epoch)

        # Nytt: logg train confusion matrix som figur i TensorBoard
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(tr_cmat, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix - Train'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(tr_cmat.shape[0]):
            for j in range(tr_cmat.shape[1]):
                ax.text(j, i, int(tr_cmat[i, j]), ha="center", va="center")
        fig.tight_layout()
        summary_writer.add_figure('ConfusionMatrix/train', fig, global_step=epoch)
        plt.close(fig)

        vl_loss, vl_acc, vl_precision, vl_recall, vl_f1, vl_cmat = validate(
            model, val_loader, criterion,
            [acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix], device,
            epoch=epoch,
            oil_drum_label=oil_drum_label,
            save_dir=oil_drum_tracking_dir
        )

        summary_writer.add_scalar('Loss/validation', vl_loss, global_step=epoch)
        summary_writer.add_scalar('Accuracy/validation', vl_acc, global_step=epoch)
        summary_writer.add_scalar('Precision/validation', vl_precision, global_step=epoch)
        summary_writer.add_scalar('Recall/validation', vl_recall, global_step=epoch)
        summary_writer.add_scalar('F1/validation', vl_f1, global_step=epoch)

        # Nytt: logg validation confusion matrix som figur i TensorBoard
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(vl_cmat, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix - Validation'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(vl_cmat.shape[0]):
            for j in range(vl_cmat.shape[1]):
                ax.text(j, i, int(vl_cmat[i, j]), ha="center", va="center")
        fig.tight_layout()
        summary_writer.add_figure('ConfusionMatrix/validation', fig, global_step=epoch)
        plt.close(fig)

        if (vl_loss < best_vl_loss):
            torch.save(model.state_dict(), 'logs/exp1.pth')
            best_vl_loss=vl_loss
        # dist.barrier()
        #break

    model.load_state_dict(torch.load(PATH, map_location=device))
    run_feature_space_analysis(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=feature_analysis_dir,
        oil_drum_label=oil_drum_label,
        clutter_label=3,
        top_k=5
    )

    print('Done')



def train(model, train_loader, optimizer, criterion, metrics,device, soft_label_dict=None):   #removed triplet_loss_func

    acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix = metrics

    model.train()
    running_loss = 0.0
        #num_samp=np.zeros((4))
    #feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    #feature_extractor.train()



    '''        
    for i, (images, labels, indexes) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
    '''
    for i, (images, labels, indexes) in enumerate(train_loader):
            images, labels, indexes = oversample_oil_drum_batch(
                images, labels, indexes, oil_label=0, factor=100
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)


            #print('shapes', rank, ' ', images.shape, ' ', labels.shape)
            #print('rank:', rank, ' index:', indexes)
            #num_samp[rank]=num_samp[rank]+np.shape(labels.cpu().numpy())[0]

            #print('rank:', rank, ' index:', i)
            #continue
            optimizer.zero_grad()
            #continue

            #outputs = model(images)   # the model output (logits) is directly used as embeddings for metric learning
            #embeddings = outputs      # old embedding setup, used logits 

            #feature_maps = feature_extractor(images)
            #embeddings = torch.flatten(feature_maps, 1)
            #outputs = model.fc(embeddings)

            outputs = model(images)   # standard classification output

            #print('outputs ', outputs.shape, ' labels ', labels.shape)
            #continue
            acc=acc_metric(outputs,labels)
            precision=precision_metric(outputs,labels)
            recall=recall_metric(outputs,labels)
            f1=f1_metric(outputs,labels)
            confmat=conf_matrix(outputs,labels)

            #loss = criterion(outputs, labels)    # Original loss

            if soft_label_dict is not None:
                soft_targets = []
                for idx, lbl in zip(indexes.cpu().numpy(), labels.cpu().numpy()):
                    if int(idx) in soft_label_dict:
                        soft_targets.append(soft_label_dict[int(idx)])
                    else:
                        one_hot = np.zeros(4)
                        one_hot[int(lbl)] = 1.0
                        soft_targets.append(one_hot)

                soft_targets = torch.tensor(np.array(soft_targets), dtype=torch.float32).to(device)

                log_probs = torch.log_softmax(outputs, dim=1)
                classification_loss = -(soft_targets * log_probs).sum(dim=1).mean()
            else:
                classification_loss = criterion(outputs, labels)

            #classification_loss = criterion(outputs, labels)   # For composite loss
            #metric_loss = triplet_loss_func(embeddings, labels)
            #loss = classification_loss + metric_loss     # For composite loss
            #loss = metric_loss     # Only metric loss 
            loss = classification_loss  # Soft-label classification loss

            #continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 :
                print(f" Step [{i}] Loss: {loss.item():.4f}")

    ave_running_loss = torch.tensor(running_loss / (i + 1)).to(device)
        #dist.all_reduce(ave_running_loss, op=dist.ReduceOp.AVG)

        ## Compute the metrics
    accuracy = acc_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
        #dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)

    confmat=conf_matrix.compute()
        #dist.all_reduce(confmat, op=dist.ReduceOp.SUM)

        #if rank == 0:
    print(f"Epoch  finished. Avg ave Loss: {ave_running_loss.item():.4f}")
    print("Accuracy:", accuracy.item())
    print("Precision:", precision.item())
    print("Recall:", recall.item())
    print("F1 Score:", f1.item())
    print("Confusion Matrix", confmat.cpu().numpy())
        #print(f"\tEpoch {epoch} finished. Avg Loss: {running_loss / (i + 1):.4f}")


        ## Reset Metrics
    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    conf_matrix.reset()

    return ave_running_loss.item(), accuracy.item(), precision.item(), recall.item(), f1.item(), confmat.cpu().numpy()


def validate(model, val_loader, criterion, metrics,device, epoch=None, oil_drum_label=0, save_dir=None):
    acc_metric, precision_metric, recall_metric, f1_metric, conf_matrix = metrics

    model.eval()
    running_loss = 0.0
    num_samp = np.zeros((4))
    oil_drum_rows = []

    print('start validation')


    with  torch.no_grad():

        for i, (images, labels, indexes) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            # softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1)

            acc = acc_metric(outputs, labels)
            precision = precision_metric(outputs, labels)
            recall = recall_metric(outputs, labels)
            f1 = f1_metric(outputs, labels)
            confmat = conf_matrix(outputs, labels)
            loss = criterion(outputs, labels)

            oil_drum_mask = (labels == oil_drum_label)
            if oil_drum_mask.any():
                oil_drum_mask_cpu = oil_drum_mask.detach().cpu()
                oil_drum_indexes = indexes[oil_drum_mask_cpu].detach().cpu().numpy()
                oil_drum_predictions = predicted_labels[oil_drum_mask].detach().cpu().numpy()
                
                oil_drum_probs = probs[oil_drum_mask].detach().cpu().numpy()
                oil_drum_true = labels[oil_drum_mask].detach().cpu().numpy()

                for sample_index, true_label, predicted_label, prob in zip(
                    oil_drum_indexes, oil_drum_true, oil_drum_predictions, oil_drum_probs
                ):
                    oil_drum_rows.append([
                        epoch,
                        int(np.asarray(sample_index).item()),
                        int(true_label),
                        int(predicted_label),
                        float(prob[0]),  # prob_class_0 (oil drum)
                        float(prob[1]),
                        float(prob[2]),
                        float(prob[3])
                    ])

            running_loss += loss.item()

            if i % 100 == 0 :
                print(f"\tStep [{i}] Loss: {loss.item():.4f}")


    if save_dir is not None and epoch is not None:
        csv_path = os.path.join(save_dir, f"oil_drum_predictions_epoch_{epoch:03d}.csv")
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "epoch",
                "index",
                "true_label",
                "predicted_label",
                "prob_class_0",
                "prob_class_1",
                "prob_class_2",
                "prob_class_3"
            ])
            writer.writerows(oil_drum_rows)

    ave_running_loss = torch.tensor(running_loss / (i + 1)).to(device)

    ## Compute the metrics. compute operation performs all_reduce itself.
    accuracy = acc_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    confmat = conf_matrix.compute()

    print(f"Epoch validation finished. Avg ave Loss: {ave_running_loss.item():.4f}")
    print("Accuracy (validation):", accuracy.item())
    print("Precision (validation):", precision.item())
    print("Recall (validation):", recall.item())
    print("F1 Score (validation):", f1.item())
    print("Confusion Matrix (validation)", confmat.cpu().numpy())

    ## Reset Metrics
    acc_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    conf_matrix.reset()

    return ave_running_loss.item(), accuracy.item(), precision.item(), recall.item(), f1.item(), confmat.cpu().numpy()


def main():
    # Configuration
    TFRECORD_PATTERN_TRN = "/projects/ec12/ec-smibrahi/Z-Dataset/tf_data/dataset_2/noaug_500px_train_i*.tfrecord" # Path to your files
    TFRECORD_PATTERN_VAL = "/projects/ec12/ec-smibrahi/Z-Dataset/tf_data/dataset_2/noaug_500px_validation_i*.tfrecord"
    WORLD_SIZE = torch.cuda.device_count()
    
    if WORLD_SIZE < 1:
        print("No GPUs found.")
        #return

    if WORLD_SIZE > 4:
        WORLD_SIZE = 1
        
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA (GPU) is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.") 

    print(f"Starting training on {WORLD_SIZE} GPUs...")
    
    # Spawn processes (one per GPU)
    '''mp.spawn(
        train_fn,
        args=(WORLD_SIZE, TFRECORD_PATTERN, INDEX_PATTERN),
        nprocs=WORLD_SIZE,
        join=True
    )'''

    if not os.path.exists("logs"):
        os.makedirs("logs")
    train_fn(device, TFRECORD_PATTERN_TRN, TFRECORD_PATTERN_VAL, epochs=25)

if __name__ == "__main__":
    main()