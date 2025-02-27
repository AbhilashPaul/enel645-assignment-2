import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import DistilBertModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#print(f"Is GPU available? {torch.cuda.is_available()}")  # Should return True
#print(f"Device count: {torch.cuda.device_count()}")  # Should return at least 1


# Constants
DATA_DIR = r"/work/TALC/enel645_2025w/garbage_data"
TRAIN_DIR = os.path.join(DATA_DIR, "CVPR_2024_dataset_Train")
VAL_DIR = os.path.join(DATA_DIR, "CVPR_2024_dataset_Val")
TEST_DIR = os.path.join(DATA_DIR, "CVPR_2024_dataset_Test")
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 15
CLASS_NAMES = ['Blue', 'Black', 'Green', 'TTR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
def get_transforms(phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ])

# Custom dataset class
class CustomImageDatasetWithDescription(datasets.ImageFolder):
    def _extract_text_description(self, file_name: str) -> str:
        clean_name = re.sub(r'_\d+|(\.png|\.jpg|\.jpeg)$', '', file_name, flags=re.IGNORECASE)
        return clean_name.replace('_', ' ').strip()

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        file_name = os.path.basename(path)
        text_description = self._extract_text_description(file_name)
        return {"image": image, "label": label, "description": text_description}

# Prepare dataloaders
def prepare_dataloaders():
    datasets_dict = {
        "train": CustomImageDatasetWithDescription(TRAIN_DIR, transform=get_transforms("train")),
        "val": CustomImageDatasetWithDescription(VAL_DIR, transform=get_transforms("val")),
        "test": CustomImageDatasetWithDescription(TEST_DIR, transform=get_transforms("test")),
    }
    dataloaders_dict = {
        phase: DataLoader(datasets_dict[phase], batch_size=BATCH_SIZE, shuffle=(phase == "train"), num_workers=1)
        for phase in ["train", "val", "test"]
    }
    return dataloaders_dict

# Tokenizer Manager for management of tokenization logic
class TokenizerManager:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None  # Lazy initialization

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def tokenize(self, descriptions: list[str], device: torch.device):
        encoding = self.tokenizer(
            descriptions,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].to(device),
            "attention_mask": encoding["attention_mask"].to(device),
        }

# Image Feature extractor model
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# Text Feature extractor model
class DistilBERTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]

# Garbage classifier model
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.image_extractor = ResNet50FeatureExtractor()
        self.text_extractor = DistilBERTFeatureExtractor()
        
        self.img_branch = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.text_branch = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, image, input_ids, attention_mask):
        img_features = F.normalize(self.img_branch(self.image_extractor(image)), p=2)
        text_features = F.normalize(self.text_branch(self.text_extractor(input_ids=input_ids,
                                                                         attention_mask=attention_mask)), p=2)
        combined_features = torch.cat((img_features, text_features), dim=1)
        return self.classifier(combined_features)

# Training function
def train_model(model, dataloaders, criterion, optimizer):
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Process batches
            for batch in dataloaders[phase]:
                tokenizer_manager = TokenizerManager(model_name="distilbert-base-uncased", max_length=128)
                # Load data
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                descriptions = batch["description"]

                tokens = tokenizer_manager.tokenize(descriptions=descriptions, device=DEVICE)
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")
                print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    return model

# Prediction function
def predict(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            tokenizer_manager = TokenizerManager(model_name="distilbert-base-uncased", max_length=128)

            # Load data
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            descriptions = batch["description"]

            tokens = tokenizer_manager.tokenize(descriptions=descriptions, device=DEVICE)
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


if __name__ == "__main__":
    
    # Initialize model and other components
    model = GarbageClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("Loading images..")
    dataloaders_dict = prepare_dataloaders()

    print("Starting training..")
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders_dict,
        criterion=criterion,
        optimizer=optimizer,
    )

    # model evaluation
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    predictions_test_set, true_labels_test_set = predict(
        model=model,
        dataloader=dataloaders_dict["test"],
    )
    # print evaluation metrics
    print(f"Accuracy: {accuracy_score(true_labels_test_set, predictions_test_set):.4f}")
    print(classification_report(true_labels_test_set, predictions_test_set, target_names=CLASS_NAMES))

    # plot confusion matrix
    cmatrix_test_set = confusion_matrix(true_labels_test_set,
                                        predictions_test_set)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cmatrix_test_set,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig('confusion_martrix.png', bbox_inches='tight')
    plt.show()
