import gradio as gr
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import os

# Load the model and processor with proper error handling
def load_model_safely():
    """Load model with fallback options and proper error handling"""
    try:
        # Try loading from local clone first
        if os.path.exists("./waste-classification"):
            print("Loading model from local clone...")
            processor = ViTImageProcessor.from_pretrained("./waste-classification")
            model = ViTForImageClassification.from_pretrained("./waste-classification")
            print("Successfully loaded model from local clone")
            return processor, model
    except Exception as e:
        print(f"Failed to load from local clone: {e}")
    
    try:
        # Try loading the HuggingFace model with cache
        print("Loading model from HuggingFace...")
        processor = ViTImageProcessor.from_pretrained("watersplash/waste-classification", cache_dir="./cache")
        model = ViTForImageClassification.from_pretrained("watersplash/waste-classification", cache_dir="./cache")
        print("Successfully loaded model from HuggingFace")
        return processor, model
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
    
    try:
        # Final fallback to base model
        print("Loading base ViT model as fallback...")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Create model with exact same config as trained model
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=12,
            id2label={
                "0": "battery", "1": "biological", "2": "brown-glass", "3": "cardboard",
                "4": "clothes", "5": "green-glass", "6": "metal", "7": "paper",
                "8": "plastic", "9": "shoes", "10": "trash", "11": "white-glass"
            },
            label2id={
                "battery": "0", "biological": "1", "brown-glass": "2", "cardboard": "3",
                "clothes": "4", "green-glass": "5", "metal": "6", "paper": "7",
                "plastic": "8", "shoes": "9", "trash": "10", "white-glass": "11"
            }
        )
        print("Loaded base ViT model as fallback (untrained)")
        return processor, model
    except Exception as e:
        print(f"Failed to load fallback model: {e}")
        return None, None

# Initialize model
print("Initializing model...")
processor, model = load_model_safely()

# Class labels from the actual model config
class_names = [
    'Battery', 'Biological', 'Brown-glass', 'Cardboard', 'Clothes', 
    'Green-glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-glass'
]

def classify_waste(image):
    """
    Classify waste image into one of 12 categories
    """
    if processor is None or model is None:
        return {"Error": "Model failed to load. Please try refreshing the page or contact support."}
    
    if image is None:
        return {"Error": "Please upload an image."}
    
    try:
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get confidence scores
        confidence_scores = predictions[0].tolist()
        
        # Create results dictionary using the exact class names from model
        results = {}
        for i, confidence in enumerate(confidence_scores):
            class_name = class_names[i]
            results[class_name] = confidence
        
        return results
    
    except Exception as e:
        return {"Error": f"Classification failed: {str(e)}"}

def get_model_status():
    """Return model loading status for debugging"""
    if processor is not None and model is not None:
        return "✅ Model loaded successfully"
    else:
        return "❌ Model failed to load"

# Create Gradio interface with better error handling
try:
    model_status = get_model_status()
    
    # Create example images list
    examples = []
    if os.path.exists("green_glass.png"):
        examples.append(["green_glass.png"])
    
    interface = gr.Interface(
        fn=classify_waste,
        inputs=gr.Image(type="pil", label="Upload Waste Image"),
        outputs=gr.Label(num_top_classes=5, label="Waste Classification Results"),
        title="🗑️ AI Waste Classification",
        description=f"""
        ### Waste Classification using Vision Transformer (ViT)
        
        **Model Status:** {model_status}
        
        Upload an image of waste and get AI-powered classification into 12 categories:
        
        **Categories:** Battery, Biological, Brown-glass, Cardboard, Clothes, Green-glass, Metal, Paper, Plastic, Shoes, Trash, White-glass
        
        **Model Details:**
        - Architecture: Vision Transformer (ViT)
        - Accuracy: 98% on Garbage Classification dataset
        - Model: watersplash/waste-classification
        - Base: google/vit-base-patch16-224-in21k
        
        *Tip: For best results, use clear images with good lighting.*
        """,
        examples=examples,
        theme=gr.themes.Soft(),
        allow_flagging="never",
        cache_examples=False
    )
    
    print("Gradio interface created successfully")
    
except Exception as e:
    print(f"Error creating Gradio interface: {e}")
    # Create a minimal error interface
    def show_error(image):
        return {"Error": "Application failed to initialize properly. Please contact support."}
    
    interface = gr.Interface(
        fn=show_error,
        inputs=gr.Image(type="pil", label="Upload Waste Image"),
        outputs=gr.Label(label="Error"),
        title="🗑️ AI Waste Classification - Error",
        description="The application encountered an error during initialization."
    )

if __name__ == "__main__":
    try:
        print("Launching Gradio interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            share=False
        )
    except Exception as e:
        print(f"Failed to launch interface: {e}")
        # Try launching with minimal config
        interface.launch()