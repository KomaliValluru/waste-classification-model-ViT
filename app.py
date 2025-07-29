import gradio as gr
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import os

# Load the model and processor with proper error handling
model_name = "watersplash/waste-classification"

def load_model_safely():
    """Load model with fallback options and proper error handling"""
    try:
        # Try loading the custom model
        processor = ViTImageProcessor.from_pretrained(model_name, cache_dir="./cache")
        model = ViTForImageClassification.from_pretrained(model_name, cache_dir="./cache")
        print(f"Successfully loaded model: {model_name}")
        return processor, model
    except Exception as e:
        print(f"Failed to load custom model: {e}")
        try:
            # Fallback to base model with local cache
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir="./cache")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=12, cache_dir="./cache")
            print("Loaded base ViT model as fallback")
            return processor, model
        except Exception as e2:
            print(f"Failed to load fallback model: {e2}")
            return None, None

# Initialize model
processor, model = load_model_safely()

# Class labels
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
    
    try:
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get confidence scores
        confidence_scores = predictions[0].tolist()
        
        # Create results dictionary
        results = {}
        for i, (class_name, confidence) in enumerate(zip(class_names, confidence_scores)):
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
    # Check if model loaded successfully before creating interface
    model_status = get_model_status()
    
    interface = gr.Interface(
        fn=classify_waste,
        inputs=gr.Image(type="pil", label="Upload Waste Image"),
        outputs=gr.Label(num_top_classes=5, label="Waste Classification Results"),
        title="🗑️ AI Waste Classification",
        description=f"""
        ### Waste Classification using Vision Transformer (ViT)
        
        **Model Status:** {model_status}
        
        Upload an image of waste and get AI-powered classification into 12 categories:
        **Battery, Biological, Brown-glass, Cardboard, Clothes, Green-glass, Metal, Paper, Plastic, Shoes, Trash, White-glass**
        
        **Model Details:**
        - Architecture: Vision Transformer (ViT)
        - Accuracy: 98% on Garbage Classification dataset
        - Model: watersplash/waste-classification
        
        *If you encounter errors, please try refreshing the page.*
        """,
        examples=[
            ["green_glass.png"]
        ] if os.path.exists("green_glass.png") else [],
        theme=gr.themes.Soft(),
        allow_flagging="never"
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
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to launch interface: {e}")
        # Try launching with minimal config
        interface.launch()