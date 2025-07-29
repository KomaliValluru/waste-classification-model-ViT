"""
Updated training script with fixes for Hugging Face deployment
"""
import torch
import math
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, AdamW, get_scheduler
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import numpy as np

# Configuration
class Config:
    model_name = "google/vit-base-patch16-224-in21k"
    num_epochs = 10
    learning_rate = 5e-5
    batch_size = 16
    val_split = 0.2
    output_model_name = "komalali/waste-classification-ViT"

def prepare_data(train_path):
    """Prepare training and validation datasets"""
    ds = ImageFolder(train_path)
    
    # Create label mappings
    label2id = {class_name: str(i) for i, class_name in enumerate(ds.classes)}
    id2label = {str(i): class_name for i, class_name in enumerate(ds.classes)}
    
    # Split dataset
    indices = torch.randperm(len(ds)).tolist()
    n_val = math.floor(len(indices) * Config.val_split)
    train_ds = torch.utils.data.Subset(ds, indices[:-n_val])
    val_ds = torch.utils.data.Subset(ds, indices[-n_val:])
    
    return train_ds, val_ds, label2id, id2label, ds.classes

class ImageClassificationCollator:
    """Updated collator using ViTImageProcessor"""
    def __init__(self, processor):
        self.processor = processor
 
    def __call__(self, batch):
        encodings = self.processor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings

def train_model(train_path, push_to_hub=False):
    """Train the waste classification model"""
    
    # Prepare data
    train_ds, val_ds, label2id, id2label, classes = prepare_data(train_path)
    
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Initialize processor and model
    processor = ViTImageProcessor.from_pretrained(Config.model_name)
    model = ViTForImageClassification.from_pretrained(
        Config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    )
    
    # Create data loaders
    collator = ImageClassificationCollator(processor)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, 
                             collate_fn=collator, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, 
                           collate_fn=collator, num_workers=2)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    num_training_steps = Config.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, 
        num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    
    for epoch in range(Config.num_epochs):
        print(f"\\nEpoch {epoch + 1}/{Config.num_epochs}")
        epoch_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            accuracy = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Final evaluation
    final_accuracy = evaluate_model(model, val_loader, device)
    print(f"\\nFinal Validation Accuracy: {final_accuracy:.4f}")
    
    # Save model
    if push_to_hub:
        model.push_to_hub(Config.output_model_name)
        processor.push_to_hub(Config.output_model_name)
        print(f"Model pushed to Hugging Face: {Config.output_model_name}")
    else:
        model.save_pretrained("./waste_classification_model")
        processor.save_pretrained("./waste_classification_model")
        print("Model saved locally to ./waste_classification_model")
    
    return model, processor, final_accuracy

def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    model.train()
    return correct / total

if __name__ == "__main__":
    # Example usage
    train_path = '/kaggle/input/garbage-classification/garbage_classification'
    
    # Train model (set push_to_hub=True to upload to Hugging Face)
    model, processor, accuracy = train_model(train_path, push_to_hub=False)
    print(f"Training completed! Final accuracy: {accuracy:.4f}")