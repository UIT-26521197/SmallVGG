from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.conv import Conv
from src.dense import Dense
from src.flatten import Flatten
from src.maxpool import MaxPool
from src.relu import ReLU
from src.SGD import SGD
from src.softmax import SoftmaxEntropyLoss


ORIGINAL_CLASSES = (0, 1, 6, 7, 8)
CLASS_TO_NEW_LABEL = {label: index for index, label in enumerate(ORIGINAL_CLASSES)}
DATA_ROOT = Path("./dataset")
SPLIT_SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.01


class VGG7:
    def __init__(self, num_classes=5):
        self.conv1_1 = Conv(3, 64, KernelSize=3, Stride=1, Padding=1)
        self.relu1_1 = ReLU()
        self.conv1_2 = Conv(64, 64, KernelSize=3, Stride=1, Padding=1)
        self.relu1_2 = ReLU()
        self.pool1 = MaxPool(KernelSize=2, Stride=2)

        self.conv2_1 = Conv(64, 128, KernelSize=3, Stride=1, Padding=1)
        self.relu2_1 = ReLU()
        self.conv2_2 = Conv(128, 128, KernelSize=3, Stride=1, Padding=1)
        self.relu2_2 = ReLU()
        self.pool2 = MaxPool(KernelSize=2, Stride=2)

        self.conv3_1 = Conv(128, 256, KernelSize=3, Stride=1, Padding=1)
        self.relu3_1 = ReLU()
        self.conv3_2 = Conv(256, 256, KernelSize=3, Stride=1, Padding=1)
        self.relu3_2 = ReLU()
        self.pool3 = MaxPool(KernelSize=2, Stride=2)

        self.flatten = Flatten()
        self.classifier = Dense(256 * 4 * 4, num_classes)

        self.layers = [
            self.conv1_1,
            self.relu1_1,
            self.conv1_2,
            self.relu1_2,
            self.pool1,
            self.conv2_1,
            self.relu2_1,
            self.conv2_2,
            self.relu2_2,
            self.pool2,
            self.conv3_1,
            self.relu3_1,
            self.conv3_2,
            self.relu3_2,
            self.pool3,
            self.flatten,
            self.classifier,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    __call__ = forward


class RemappedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, original_label = self.dataset[self.indices[index]]
        return image, CLASS_TO_NEW_LABEL[original_label]


def build_datasets(root=DATA_ROOT, seed=SPLIT_SEED):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    train_base = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_base = CIFAR10(root=root, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []

    for original_label in ORIGINAL_CLASSES:
        class_indices = [
            index
            for index, label in enumerate(train_base.targets)
            if label == original_label
        ]
        permutation = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[index] for index in permutation]
        train_indices.extend(shuffled[:4000])
        validation_indices.extend(shuffled[4000:5000])

    test_indices = [
        index
        for index, label in enumerate(test_base.targets)
        if label in CLASS_TO_NEW_LABEL
    ]

    train_dataset = RemappedSubset(train_base, train_indices)
    validation_dataset = RemappedSubset(train_base, validation_indices)
    test_dataset = RemappedSubset(test_base, test_indices)
    return train_dataset, validation_dataset, test_dataset


def move_model_to_device(model, device):
    for layer in model.layers:
        for name in ("Weight", "Bias"):
            parameter = getattr(layer, name, None)
            if isinstance(parameter, torch.Tensor):
                moved = parameter.detach().to(device)
                moved.requires_grad_()
                setattr(layer, name, moved)


def get_trainable_layers(model):
    return [
        layer
        for layer in model.layers
        if hasattr(layer, "parameters") and len(layer.parameters()) > 0
    ]


def clear_gradients(trainable_layers):
    for layer in trainable_layers:
        layer.zero_grad()


def train_one_epoch(model, data_loader, criterion, optimizer, trainable_layers, device):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        clear_gradients(trainable_layers)
        logits = model.forward(images)
        loss = criterion.forward(logits, labels)
        loss.backward()

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_loss += loss.item() * labels.shape[0]
        total_samples += labels.shape[0]

        for layer in trainable_layers:
            optimizer.update(layer)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, data_loader, criterion, device):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward(images)
            loss = criterion.forward(logits, labels)
            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_loss += loss.item() * labels.shape[0]
            total_samples += labels.shape[0]

    return total_loss / total_samples, total_correct / total_samples


def run_training():
    train_dataset, validation_dataset, _ = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = VGG7(num_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move_model_to_device(model, device)

    trainable_layers = get_trainable_layers(model)
    criterion = SoftmaxEntropyLoss()
    optimizer = SGD(learningRate=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            trainable_layers,
            device,
        )
        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
        )
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    run_training()
