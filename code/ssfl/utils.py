import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import sys
import os
from torchvision import datasets, transforms



import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split

from torch.utils.data import Subset, random_split 
import json
from datasets import load_dataset as hf_load_dataset  # Import the huggingface loader
from transformers import BertTokenizer
import requests

from torch.utils.data import Dataset, ConcatDataset






random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True



def save_model(model, path):
    torch.save(model.state_dict(), path)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.0 * correct.float() / preds.shape[0]
    return acc.item()

def create_model(model_name: str, num_classes: int, in_channels=3):
    from torchvision import models
    from ssfl.model_splitter import CNNBase
    if model_name.lower() == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name.lower() == 'cnn':
        return CNNBase(num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")



class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data['dx'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.image_dir}/{self.data.loc[idx, 'image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[self.data.loc[idx, 'dx']]
        if self.transform:
            image = self.transform(image)
        return image, label

class LeafShakespeareDataset(Dataset):
    """
    A PyTorch Dataset for the Leaf-preprocessed Shakespeare dataset.
    This dataset is designed for next-character prediction.
    """
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.users = data['users']
        self.user_data = data['user_data']
        self.num_samples = data['num_samples']

        # Consolidate all text to build a vocabulary
        all_text = ""
        for user in self.users:
            all_text += "".join(self.user_data[user]['x'])
            all_text += "".join(self.user_data[user]['y'])

        self.vocab = sorted(list(set(all_text)))
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab)}

        # Flatten the data into a single list of (input_sequence, target_character)
        self.all_samples = []
        for user in self.users:
            for i in range(len(self.user_data[user]['x'])):
                self.all_samples.append({
                    'x': self.user_data[user]['x'][i],
                    'y': self.user_data[user]['y'][i]
                })

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        input_seq = sample['x']
        target_char = sample['y']

        # Encode input and target characters
        input_tensor = torch.tensor([self.char_to_int[char] for char in input_seq], dtype=torch.long)
        target_tensor = torch.tensor(self.char_to_int[target_char], dtype=torch.long)

        return input_tensor, target_tensor

def load_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()



    if dataset_name == "pacs_old":
        print("--- Loading PACS dataset from Hugging Face Hub... ---")
        
        image_size = 224 
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        raw_pacs_dataset = hf_load_dataset("flwrlabs/pacs", split='train')

        # 1. Define the domains to use as clients
        client_domains = ["art_painting", "cartoon", "photo", "sketch"]
        print(f"--- Filtering dataset into domains: {client_domains} ---")

        # 2. Filter the full dataset to create one dataset per domain
        domain_hf_datasets = [
            raw_pacs_dataset.filter(lambda example: example["domain"] == domain)
            for domain in client_domains
        ]

        # This wrapper class is where we apply the fix.
        class PyTorchPACSDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform
                # Create the .targets attribute for compatibility with your partitioner
                self.targets = [item['label'] for item in hf_dataset]

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                # =================== START OF THE FIX ===================
                # The DataLoader can pass idx as a numpy.int64, but hf_dataset requires a standard int.
                # We explicitly cast the index to a Python int before using it.
                item = self.hf_dataset[int(idx)]
                # =================== END OF THE FIX =====================
                
                image = self.transform(item['image'].convert("RGB"))
                label = item['label']
                return image, label

        train_datasets_list = [PyTorchPACSDataset(ds, transform) for ds in domain_hf_datasets]
        test_subsets = []
        for ds in train_datasets_list:
            num_test_samples = int(0.1 * len(ds))
            if num_test_samples == 0 and len(ds) > 0:
                num_test_samples = 1
            num_train_samples = len(ds) - num_test_samples
            if num_train_samples > 0 and num_test_samples > 0:
                _, test_subset = torch.utils.data.random_split(ds, [num_train_samples, num_test_samples])
                test_subsets.append(test_subset)
        
        test_dataset = ConcatDataset(test_subsets) if test_subsets else None


        # full_dataset = PyTorchPACSDataset(raw_pacs_dataset, transform)

        # train_size = int(0.9 * len(full_dataset))
        # test_size = len(full_dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
        return train_datasets_list, test_dataset, 7, image_size, 3

    elif dataset_name == "pacs":
        print("--- Loading PACS dataset from Hugging Face Hub... ---")
        
        image_size = 224 
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        raw_pacs_dataset = hf_load_dataset("flwrlabs/pacs", split='train')

        # 1. Define all available domains
        all_domains = ["art_painting", "cartoon", "photo", "sketch"]
        
        # --- MODIFICATION START ---
        # 2. Designate one domain as the TEST DOMAIN for cross-domain evaluation
        #    You can choose any domain here, e.g., 'sketch', 'cartoon', etc.
        #    For a full evaluation, you might loop through each domain as the test set.
        test_domain = "photo" # Example: 'sketch' will be the test set for this run
        
        # 3. Define client training domains (all domains EXCEPT the test_domain)
        client_training_domains = [d for d in all_domains if d != test_domain]
        print(f"--- Training clients on domains: {client_training_domains} ---")
        print(f"--- Testing on held-out domain: {test_domain} ---")

        # 4. Filter the full dataset to create training datasets per client domain
        domain_hf_train_datasets = [
            raw_pacs_dataset.filter(lambda example: example["domain"] == domain)
            for domain in client_training_domains
        ]
        
        # 5. Filter the full dataset to create the global test dataset from the held-out domain
        hf_test_domain_dataset = raw_pacs_dataset.filter(lambda example: example["domain"] == test_domain)

        # This wrapper class is where we apply the fix.
        class PyTorchPACSDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self.targets = [item['label'] for item in hf_dataset]

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                item = self.hf_dataset[int(idx)]
                image = self.transform(item['image'].convert("RGB"))
                label = item['label']
                return image, label

        # 6. Create PyTorch-compatible datasets for training clients
        train_datasets_list = [PyTorchPACSDataset(ds, transform) for ds in domain_hf_train_datasets]
        
        # 7. Create the global test dataset from the held-out domain
        test_dataset = PyTorchPACSDataset(hf_test_domain_dataset, transform)

        # Ensure test_dataset is not empty if the designated test domain has no data
        if len(test_dataset) == 0:
            print(f"Warning: The designated test domain '{test_domain}' yielded 0 samples. Check data integrity.")
            # You might want to raise an error or select a different test domain here.
            test_dataset = None # Or provide a dummy empty dataset

        # --- END MODIFICATION ---

        # 8. Set metadata and return
        num_classes = 7 # PACS has 7 classes
        # image_size and in_channels are already set above
        
        # Return train_datasets_list (now has 3 domains), and the test_dataset (1 held-out domain)
        return train_datasets_list, test_dataset, num_classes, image_size, 3

    elif dataset_name == "officehome":
        print("--- Loading Office-Home dataset from Hugging Face Hub... ---")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("--- Loading all domains... ---")
        full_hf_dataset = hf_load_dataset("flwrlabs/office-home", split='train')

        # 1. Define all available domains
        all_domains = ["Art", "Clipart", "Product", "Real World"] # Ensure "Real World" is correct
        
        # --- MODIFICATION START ---
        # 2. Designate one domain as the TEST DOMAIN for cross-domain evaluation
        test_domain = "Real World" # Example: 'Real World' will be the test set for this run
        
        # 3. Define client training domains (all domains EXCEPT the test_domain)
        client_training_domains = [d for d in all_domains if d != test_domain]
        print(f"--- Training clients on domains: {client_training_domains} ---")
        print(f"--- Testing on held-out domain: {test_domain} ---")

        # 4. Filter the full dataset to create training datasets per client domain
        domain_hf_train_datasets = [
            full_hf_dataset.filter(lambda example: example["domain"] == domain)
            for domain in client_training_domains
        ]
        
        # 5. Filter the full dataset to create the global test dataset from the held-out domain
        hf_test_domain_dataset = full_hf_dataset.filter(lambda example: example["domain"] == test_domain)

        # 6. Create your PyTorch wrapper class (unchanged and correct)
        class PyTorchOfficeHomeDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self.targets = [item['label'] for item in hf_dataset]
            def __len__(self):
                return len(self.hf_dataset)
            def __getitem__(self, idx):
                item = self.hf_dataset[int(idx)]
                image = self.transform(item['image'].convert("RGB"))
                label = item['label']
                return image, label

        # 7. Create PyTorch-compatible datasets for training clients
        train_datasets_list = [PyTorchOfficeHomeDataset(ds, transform) for ds in domain_hf_train_datasets]
        
        # 8. Create the global test dataset from the held-out domain
        test_dataset = PyTorchOfficeHomeDataset(hf_test_domain_dataset, transform)

        # Ensure test_dataset is not empty
        if len(test_dataset) == 0:
            print(f"Warning: The designated test domain '{test_domain}' yielded 0 samples. Check data integrity.")
            # You might want to raise an error or select a different test domain here.
            test_dataset = None # Or provide a dummy empty dataset

        # --- END MODIFICATION ---

        num_classes = 65
        image_size = 224
        in_channels = 3
        
        # Return train_datasets_list (now has 3 domains), and the test_dataset (1 held-out domain)
        return train_datasets_list, test_dataset, num_classes, image_size, in_channels
    

    elif dataset_name == "officehome_old":
        print("--- Loading Office-Home dataset from Hugging Face Hub... ---")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # =================== START OF THE FINAL FIX ===================

        # 1. Load the ENTIRE dataset under the "default" configuration
        print("--- Loading all domains... ---")
        full_hf_dataset = hf_load_dataset("flwrlabs/office-home", split='train')

        # 2. Define the domains you want to use as clients
        client_domains = ["Art", "Clipart", "Product", "Real World"]
        print(f"--- Filtering dataset into domains: {client_domains} ---")

        # 3. Filter the full dataset to create one dataset per domain
        #    The `domain` column in the dataset has string names like "Art", "Clipart", etc.
        domain_hf_datasets = [
            full_hf_dataset.filter(lambda example: example["domain"] == domain)
            for domain in client_domains
        ]

        # 4. Create your PyTorch wrapper class (this definition is unchanged and correct)
        class PyTorchOfficeHomeDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.hf_dataset = hf_dataset
                self.transform = transform
                self.targets = [item['label'] for item in hf_dataset]
            def __len__(self):
                return len(self.hf_dataset)
            def __getitem__(self, idx):
                item = self.hf_dataset[int(idx)]
                image = self.transform(item['image'].convert("RGB"))
                label = item['label']
                return image, label

        # 5. Create a list of PyTorch-compatible datasets, one for each filtered domain
        train_datasets_list = [PyTorchOfficeHomeDataset(ds, transform) for ds in domain_hf_datasets]

        # 6. Create a balanced global test set
        test_subsets = []
        for ds in train_datasets_list:
            num_test_samples = int(0.1 * len(ds))
            if num_test_samples == 0 and len(ds) > 0: # Ensure at least one test sample if possible
                num_test_samples = 1
            num_train_samples = len(ds) - num_test_samples
            if num_train_samples > 0:
                _, test_subset = torch.utils.data.random_split(ds, [num_train_samples, num_test_samples])
                test_subsets.append(test_subset)
        
        test_dataset = ConcatDataset(test_subsets)

        # 7. Set metadata and return the list of datasets
        num_classes = 65
        image_size = 224
        in_channels = 3
        
        return train_datasets_list, test_dataset, num_classes, image_size, in_channels

    elif dataset_name == "shakespeare":
        # raw_dataset = hf_load_dataset("tiny_shakespeare")['train']
        # full_text = "".join(raw_dataset['text'])

        print("--- Downloading the complete works of Shakespeare from Project Gutenberg... ---")
        url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error if the download fails
        full_text = response.text
        print(f"--- Download complete. Total characters: {len(full_text)} ---")


        # 2. Create a true character-level vocabulary
        chars = sorted(list(set(full_text)))
        vocab_size = len(chars)
        print(f"--- Shakespeare Dataset: Building a character vocabulary of size: {vocab_size} ---")
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}

        # vocab = sorted(list(set(full_text)))
        # vocab_size = len(vocab)
        # char_to_int = {ch: i for i, ch in enumerate(vocab)}
        # int_to_char = {i: ch for i, ch in enumerate(vocab)}

        # 2. Define a PyTorch Dataset for next-character prediction
        class NextCharDataset_old(Dataset):
            def __init__(self, text, tokenizer, max_length=80):
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.encoded_text = self.tokenizer.encode(
                    text,
                    add_special_tokens=False, # We handle chunks manually
                    return_tensors="pt"
                ).squeeze(0)

                # Create the .targets attribute for compatibility
                #self.targets = self.encoded_text[1:].tolist()
                self.targets = []

            def __len__(self):
                # The number of possible sequences we can create
                #return len(self.encoded_text) - self.max_length
                return len(self.encoded_text) - self.max_length - 1

            def __getitem__(self, idx):
                # The input sequence is a chunk of text tokens
                #chunk = self.encoded_text[idx : idx + self.max_length]
                input_chunk = self.encoded_text[idx : idx + self.max_length]
                # The label for loss is the very next token after the chunk
                #label = self.encoded_text[idx + self.max_length]
                target_chunk = self.encoded_text[idx + 1 : idx + self.max_length + 1]

                # BERT models expect specific dictionary keys
                return {
                    "input_ids": input_chunk,
                    "attention_mask": torch.ones_like(input_chunk), # No padding, so mask is all 1s
                    "labels": target_chunk # Shape: [max_length]
                }
        class NextCharDataset(Dataset):
            def __init__(self, text, max_length=80):
                self.max_length = max_length
                # Encode the text using our simple character map
                self.encoded_text = [char_to_int[ch] for ch in text]

            def __len__(self):
                # The number of possible sequences we can create
                return len(self.encoded_text) - self.max_length - 1

            def __getitem__(self, idx):
                # The input sequence is a chunk of text tokens
                input_chunk = torch.tensor(self.encoded_text[idx : idx + self.max_length], dtype=torch.long)
                
                # The target sequence is the same chunk, shifted one to the right
                target_chunk = torch.tensor(self.encoded_text[idx + 1 : idx + self.max_length + 1], dtype=torch.long)

                # Return a dictionary compatible with our trainer
                return {
                    "input_ids": input_chunk,
                    "attention_mask": torch.ones_like(input_chunk), # Still needed for signature compatibility
                    "labels": target_chunk
                }

        # 3. Create tokenizer and datasets
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        n = len(full_text)
        train_text = full_text[:int(n * 0.9)]
        test_text = full_text[int(n * 0.9):]

        train_dataset = NextCharDataset(train_text)
        test_dataset = NextCharDataset(test_text)

        # train_dataset = NextCharDataset(train_text, tokenizer)
        # test_dataset = NextCharDataset(test_text, tokenizer)

        sequence_length = 80
        in_channels = 1
        
        # The number of classes for shakespeare is now vocab_size
        return train_dataset, test_dataset, vocab_size, sequence_length, in_channels
    # Set default image size
    if image_size is None:
        if dataset_name == "cifar10":
            image_size = 32
        elif dataset_name == "mnist" or dataset_name == "femnist":
            image_size = 28
        elif dataset_name == "imagenet" or dataset_name == "ham10000":
            image_size = 224
        else:
            raise ValueError("Provide image_size for unknown dataset")

    #in_channels = 1 if dataset_name in ["mnist", "femnist"] else 3
    # Set native in_channels for dataset
    native_in_channels = 1 if dataset_name in ["mnist", "femnist"] else 3

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        DatasetClass = datasets.CIFAR10
        train_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform)
        test_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_channels = 3  # ResNet18 and CNN both use 3 channels for CIFAR-10

    elif dataset_name == "mnist":
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if native_in_channels == 1 else x),  # Repeat for 3-channel models
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if in_channels == 3 else x),  # Use in_channels
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) if model_name.lower() == "resnet18" else x),  # Conditional transform
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # convert grayscale to RGB
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        DatasetClass = datasets.MNIST
        train_dataset = DatasetClass(root='./data', train=True, download=True, transform=transform)
        test_dataset = DatasetClass(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        in_channels = 3  # Always return 3 channels for MNIST to support ResNet18

    elif dataset_name == "femnist":
        #from torchvision.datasets import FEMNIST

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # The model expects 3 channels, but FEMNIST is 1-channel. This repeats the channel to make it compatible.
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # Use the standard normalization for MNIST-like datasets
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.EMNIST(root='./data', split='byclass', download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='byclass', download=True, transform=transform)
        


        num_classes = 62
        # Your models are designed for 3-channel inputs, so we set this to 3.
        in_channels = 3


    elif dataset_name == "ham10000":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7630, 0.5456, 0.5702],
                                 std=[0.1409, 0.1523, 0.1695])  # HAM10000 stats or ImageNet
        ])
        full_dataset = HAM10000Dataset(
            csv_file='./data/HAM10000/HAM10000_metadata.csv',
            image_dir='./data/HAM10000/images',
            transform=transform
        )
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        num_classes = 7

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset, num_classes, image_size, in_channels



def load_dataset_old(dataset_name, transform):
    if dataset_name.lower() == "cifar10":
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")
    return train_dataset, test_dataset

def subsample_dataset(dataset, sample_fraction):
    num_samples = int(len(dataset) * sample_fraction)
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)

def partition_text_non_iid_dirichlet(dataset, num_clients, imbalance_factor, min_samples_per_client):
    """
    Partitions a dataset into non-IID subsets using a Dirichlet distribution,
    while ensuring a minimum number of samples per client.

    Args:
        dataset: The full dataset to partition.
        num_clients: The number of clients.
        imbalance_factor (float): Controls the level of non-IID-ness.
        min_samples_per_client (int): The minimum number of samples each client must have.
    
    Returns:
        A list of Subsets, one for each client.
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    # Ensure the minimum sample requirement is feasible
    if num_clients * min_samples_per_client > num_samples:
        raise ValueError("The total minimum number of samples required is greater than the dataset size.")

    # 1. Initial distribution using Dirichlet
    client_data_proportions = np.random.dirichlet(np.repeat(imbalance_factor, num_clients))
    client_sample_counts = (client_data_proportions * num_samples).astype(int)

    # =================== START OF THE NEW LOGIC ===================

    # 2. Identify clients below the minimum and the "deficit" of samples
    below_min_indices = np.where(client_sample_counts < min_samples_per_client)[0]
    deficit = np.sum(min_samples_per_client - client_sample_counts[below_min_indices])
    
    # 3. Set the clients that were below the minimum to have the minimum
    client_sample_counts[below_min_indices] = min_samples_per_client
    
    # 4. Find clients that are "donors" (have more than the minimum)
    above_min_indices = np.where(client_sample_counts > min_samples_per_client)[0]
    
    # 5. Take back the deficit from the donor clients
    # We take samples proportionally from the donors that have a surplus
    surplus_proportions = client_sample_counts[above_min_indices] - min_samples_per_client
    
    if np.sum(surplus_proportions) < deficit:
        # This is a rare edge case where donors don't have enough surplus
        # In this case, we'll just take what we can and adjust
        client_sample_counts[above_min_indices] = min_samples_per_client
    else:
        # Distribute the deficit proportionally among the donors
        reduction_proportions = surplus_proportions / np.sum(surplus_proportions)
        reduction_amounts = (reduction_proportions * deficit).astype(int)
        client_sample_counts[above_min_indices] -= reduction_amounts

    # =================== END OF THE NEW LOGIC =====================

    # Final adjustment to ensure the total number of samples is exactly correct
    # due to rounding, and add the remainder to the client with the most samples.
    diff = np.sum(client_sample_counts) - num_samples
    richest_client = np.argmax(client_sample_counts)
    client_sample_counts[richest_client] -= diff

    # Assert that the sum is correct
    assert np.sum(client_sample_counts) == num_samples, "Final sample counts do not sum up to total samples"
    assert np.all(client_sample_counts >= min_samples_per_client), "Some clients still have fewer than the minimum samples"

    # The rest of the function remains the same
    client_subsets = []
    current_idx = 0
    np.random.shuffle(indices)

    print(f"Partitioning data with Dirichlet (factor={imbalance_factor}, min_samples={min_samples_per_client}): {num_samples} samples for {num_clients} clients.")

    for i in range(num_clients):
        num_client_samples = client_sample_counts[i]
        client_indices = indices[current_idx : current_idx + num_client_samples]
        client_subsets.append(Subset(dataset, indices=client_indices))
        current_idx += num_client_samples
        #print(f"  - Client {i}: {len(client_indices)} samples")
        
    return client_subsets


def partition_text_non_iid_dirichlet_old(dataset, num_clients, imbalance_factor):
    """
    Partitions a dataset into non-IID subsets using a Dirichlet distribution.
    This is a standard method for creating realistic non-IID data distributions
    for text and other types of data in Federated Learning.

    Args:
        dataset: The full dataset to partition.
        num_clients: The number of clients.
        imbalance_factor (float): Controls the level of non-IID-ness. A smaller
                                  value creates more imbalanced (more non-IID) data.
                                  A large value approaches an IID distribution.

    Returns:
        A list of Subsets, one for each client.
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    # Ensure the minimum sample requirement is feasible
    if num_clients * min_samples_per_client > num_samples:
        raise ValueError("The total minimum number of samples required is greater than the dataset size.")

    
    # Create a matrix to hold the number of samples for each client
    client_data_proportions = np.random.dirichlet(np.repeat(imbalance_factor, num_clients))
    #client_sample_counts = (client_data_proportions * num_samples).astype(int)

    
    # Ensure every client gets at least one sample
    client_data_proportions = client_data_proportions / np.sum(client_data_proportions)
    
    client_sample_counts = (client_data_proportions * num_samples).astype(int)
    
    # Due to rounding, we might have a slight mismatch, so adjust the last client
    diff = np.sum(client_sample_counts) - num_samples
    client_sample_counts[-1] -= diff

    # Assert that the sum is correct
    assert np.sum(client_sample_counts) == num_samples, "Sample counts do not sum up to total samples"

    client_subsets = []
    current_idx = 0
    np.random.shuffle(indices) # Shuffle indices to ensure random distribution of sequences

    print(f"Partitioning data with Dirichlet (factor={imbalance_factor}): {num_samples} samples for {num_clients} clients.")

    for i in range(num_clients):
        num_client_samples = client_sample_counts[i]
        client_indices = indices[current_idx : current_idx + num_client_samples]
        client_subsets.append(Subset(dataset, client_indices))
        current_idx += num_client_samples
        print(f"  - Client {i}: {len(client_indices)} samples")
        
    return client_subsets


def partition_data_non_iid_random(subset, num_clients, imbalance_factor, min_samples_per_client):
    """
    Partitions a dataset into non-IID subsets based on class labels.
    This version is compatible with ALL dataset types by correctly unwrapping the
    torch.utils.data.Subset object first.
    """
    
    # =================== START OF THE FINAL FIX ===================

    # 1. Correctly access the REAL dataset inside the wrapper(s).
    # This loop handles cases where a Subset might be wrapped in another Subset.
    underlying_dataset = subset
    while isinstance(underlying_dataset, Subset):
        underlying_dataset = underlying_dataset.dataset
    # Now, `underlying_dataset` is the actual dataset object (e.g., CIFAR10, PACS).

    # 2. Get the indices for the specific partition we are working with.
    subset_indices = np.array(subset.indices)

    # 3. Get ALL labels from the real dataset.
    try:
        # This works for torchvision datasets AND our PyTorchPACS wrapper
        all_labels = np.array(underlying_dataset.targets)
    except AttributeError:
        # This works for raw Hugging Face datasets
        print("  - Detected raw Hugging Face dataset. Accessing labels via column name.")
        all_labels = np.array(underlying_dataset['label'])

    # 4. Get the specific labels for just the samples in our current subset.
    subset_targets = all_labels[subset_indices]

    # 5. Dynamically determine the number of classes.
    unique_classes = np.unique(subset_targets)
    num_classes = len(unique_classes)
    print(f"  - Partitioning data for {num_classes} classes.")

    # =================== END OF THE FINAL FIX =====================

    # The rest of your function's logic now works correctly with the right labels.
    class_indices = {cls: np.where(subset_targets == cls)[0] for cls in unique_classes}
    client_data_indices = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        client_class_distribution = np.random.dirichlet(np.ones(num_classes) * imbalance_factor)
        
        for i, cls in enumerate(unique_classes):
            class_specific_indices = class_indices.get(cls, np.array([]))
            if len(class_specific_indices) > 0:
                num_samples = int(client_class_distribution[i] * len(class_specific_indices))
                if num_samples > 0:
                    chosen_indices = np.random.choice(class_specific_indices, min(num_samples, len(class_specific_indices)), replace=False)
                    client_data_indices[client_id].extend(chosen_indices)
        
        np.random.shuffle(client_data_indices[client_id])
        
        if len(client_data_indices[client_id]) < min_samples_per_client:
            needed = min_samples_per_client - len(client_data_indices[client_id])
            client_assigned_set = set(client_data_indices[client_id])
            available_pool = [idx for idx in range(len(subset_targets)) if idx not in client_assigned_set]
            additional_indices = np.random.choice(available_pool, min(needed, len(available_pool)), replace=False)
            client_data_indices[client_id].extend(additional_indices)

    # Convert the local indices (which are relative to the subset) back to global indices
    final_client_indices = {
        client_id: subset_indices[indices] for client_id, indices in client_data_indices.items()
    }
    
    # Return a list of new Subsets based on the original underlying dataset
    return [Subset(underlying_dataset, list(indices)) for indices in final_client_indices.values()]

def partition_data_non_iid_random_old(subset, num_clients, imbalance_factor, min_samples_per_client):
    full_dataset = subset.dataset
    subset_indices = subset.indices
    num_classes = 10  # CIFAR-10
    subset_targets = np.array(full_dataset.targets)[subset_indices]
    class_indices = {i: np.where(subset_targets == i)[0] for i in range(num_classes)}
    client_data_indices = {i: [] for i in range(num_clients)}

    for client_id in range(num_clients):
        client_class_distribution = np.random.dirichlet(np.ones(num_classes) * imbalance_factor)
        for cls in range(num_classes):
            num_samples = int(client_class_distribution[cls] * len(class_indices[cls]))
            if num_samples > 0:
                client_data_indices[client_id].extend(np.random.choice(class_indices[cls], num_samples, replace=False))
        np.random.shuffle(client_data_indices[client_id])
        if len(client_data_indices[client_id]) < min_samples_per_client:
            print(f"Warning: Client {client_id} has {len(client_data_indices[client_id])} samples.")
            remaining_samples_needed = min_samples_per_client - len(client_data_indices[client_id])
            available_indices = np.concatenate([class_indices[cls] for cls in range(num_classes)])
            additional_indices = np.random.choice(available_indices, remaining_samples_needed, replace=False)
            client_data_indices[client_id].extend(additional_indices)

    client_data_indices = {client_id: subset_indices[indices] for client_id, indices in client_data_indices.items()}
    return [Subset(subset.dataset, indices) for indices in client_data_indices.values()]

def create_dataloaders(subsets, batch_size, shuffle):
    return [DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=True) for subset in subsets]


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

def split_client_data(client_subset, val_split_ratio=0.2):
    """
    Splits a client's dataset (a PyTorch Subset) into training and validation.
    """
    num_total = len(client_subset)
    if num_total == 0: # Handle empty client subset
        return client_subset, Subset(client_subset.dataset, [])

    num_val = int(num_total * val_split_ratio)
    if num_val == 0 and num_total > 1 and val_split_ratio > 0: num_val = 1 # Ensure at least 1 val sample
    
    num_train = num_total - num_val
    if num_train <= 0: # If not enough data for a split
        return client_subset, Subset(client_subset.dataset, [])

    train_local_indices, val_local_indices = random_split(range(num_total), [num_train, num_val])
    
    original_dataset_train_indices = [client_subset.indices[i] for i in train_local_indices.indices]
    original_dataset_val_indices = [client_subset.indices[i] for i in val_local_indices.indices]

    return Subset(client_subset.dataset, original_dataset_train_indices), \
           Subset(client_subset.dataset, original_dataset_val_indices)

def prepare_client_dataloaders_for_hpo(
    client_data_subsets, # Output of partition_data_non_iid_random
    batch_size,
    val_split_ratio=0.2,
    num_workers=0, # Make num_workers configurable
    pin_memory=False # Make pin_memory configurable
):
    """
    Prepares client-specific training and validation DataLoaders for HPO.

    Args:
        client_data_subsets (list): List of PyTorch Subsets, one for each client.
        batch_size (int): The batch size for the DataLoaders.
        val_split_ratio (float): Fraction of a client's data for local validation.
        num_workers (int): Number of worker processes for DataLoader.
        pin_memory (bool): If True, DataLoader will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: (client_train_loaders, client_val_loaders)
               client_train_loaders: List of DataLoaders for client training.
               client_val_loaders: Dict (client_id -> DataLoader) for client validation.
    """
    client_train_loaders = []
    client_val_loaders = {} 

    for i, per_client_subset in enumerate(client_data_subsets):
        if len(per_client_subset) == 0: # Handle clients with no data
            empty_ds = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
            client_train_loaders.append(DataLoader(empty_ds, batch_size=batch_size))
            client_val_loaders[i] = DataLoader(empty_ds, batch_size=batch_size)
            continue
        
        client_local_train_data, client_local_val_data = split_client_data(
            per_client_subset, 
            val_split_ratio=val_split_ratio
        )

        train_loader = DataLoader(
            client_local_train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        client_train_loaders.append(train_loader)

        # Ensure validation loader is created even if client_local_val_data is empty after split
        if len(client_local_val_data) > 0:
            val_loader = DataLoader(
                client_local_val_data, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )
        else: # Create an empty DataLoader if validation set is empty
            empty_ds_val = torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0))
            val_loader = DataLoader(empty_ds_val, batch_size=batch_size)
            
        client_val_loaders[i] = val_loader
            
    return client_train_loaders, client_val_loaders
            
class Tee(object):
    def __init__(self, filename, mode='w'):
        self.terminal = sys.stdout
        self.log_file = open(filename, mode)
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()

    