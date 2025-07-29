import gradio as gr
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np

# Load the model and processor
model_name = "komalali/waste-classification-ViT"
try:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
except:
    # Fallback to base model if custom model not available
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=12)

# Class labels
class_names = [
    'Battery', 'Biological', 'Brown-glass', 'Cardboard', 'Clothes', 
    'Green-glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-glass'
]

def classify_waste(image):
    """
    Classify waste image into one of 12 categories
    """
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top 3 predictions
        confidence_scores = predictions[0].tolist()
        
        # Create results dictionary
        results = {}
        for i, (class_name, confidence) in enumerate(zip(class_names, confidence_scores)):
            results[class_name] = confidence
        
        return results
    
    except Exception as e:
        return {"Error": f"Classification failed: {str(e)}"}

# Create Gradio interface
interface = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=gr.Label(num_top_classes=5, label="Waste Classification Results"),
    title="🗑️ AI Waste Classification",
    description="""
    ### Waste Classification using Vision Transformer (ViT)
    
    Upload an image of waste and get AI-powered classification into 12 categories:
    **Battery, Biological, Brown-glass, Cardboard, Clothes, Green-glass, Metal, Paper, Plastic, Shoes, Trash, White-glass**
    
    **Model Details:**
    - Architecture: Vision Transformer (ViT)
    - Accuracy: 98% on Garbage Classification dataset
    - Base Model: google/vit-base-patch16-224-in21k
    """,
    examples=[
        ["green_glass.png"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()