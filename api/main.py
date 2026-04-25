from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import json 
import io
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# treatment dict
treatments = {
    "Apple___Apple_scab": "Apply fungicide sprays early in spring. Remove and destroy infected leaves.",
    "Apple___Black_rot": "Prune infected branches, apply copper-based fungicide, remove mummified fruits.",
    "Apple___Cedar_apple_rust": "Apply fungicide before infection, remove nearby cedar trees if possible.",
    "Apple___healthy": "No treatment needed. Keep monitoring regularly.",
    "Blueberry___healthy": "No treatment needed. Maintain proper watering and fertilization.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicide, ensure good air circulation.",
    "Cherry_(including_sour)___healthy": "No treatment needed. Keep monitoring regularly.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicide, rotate crops, use resistant varieties.",
    "Corn_(maize)___Common_rust_": "Apply fungicide early, use rust-resistant hybrid seeds.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant varieties, apply fungicide, practice crop rotation.",
    "Corn_(maize)___healthy": "No treatment needed. Keep monitoring regularly.",
    "Grape___Black_rot": "Remove infected parts, apply fungicide, ensure good air circulation.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood, apply fungicide, avoid water stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply copper-based fungicide, remove infected leaves.",
    "Grape___healthy": "No treatment needed. Keep monitoring regularly.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure available. Remove infected trees to prevent spread.",
    "Peach___Bacterial_spot": "Apply copper-based bactericide, avoid overhead irrigation.",
    "Peach___healthy": "No treatment needed. Keep monitoring regularly.",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericide, use disease-free seeds.",
    "Pepper,_bell___healthy": "No treatment needed. Keep monitoring regularly.",
    "Potato___Early_blight": "Apply fungicide, practice crop rotation, remove infected leaves.",
    "Potato___Late_blight": "Apply fungicide immediately, remove infected plants, avoid overhead watering.",
    "Potato___healthy": "No treatment needed. Keep monitoring regularly.",
    "Raspberry___healthy": "No treatment needed. Keep monitoring regularly.",
    "Soybean___healthy": "No treatment needed. Keep monitoring regularly.",
    "Squash___Powdery_mildew": "Apply sulfur or neem oil spray, ensure good air circulation.",
    "Strawberry___Leaf_scorch": "Remove infected leaves, apply fungicide, avoid overhead irrigation.",
    "Strawberry___healthy": "No treatment needed. Keep monitoring regularly.",
    "Tomato___Bacterial_spot": "Apply copper-based bactericide, use disease-free seeds, avoid overhead watering.",
    "Tomato___Early_blight": "Apply fungicide, remove infected leaves, practice crop rotation.",
    "Tomato___Late_blight": "Apply copper-based fungicide immediately, remove infected plants.",
    "Tomato___Leaf_Mold": "Improve air circulation, apply fungicide, reduce humidity.",
    "Tomato___Septoria_leaf_spot": "Apply fungicide, remove infected leaves, avoid overhead watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticide or neem oil, increase humidity, remove infected leaves.",
    "Tomato___Target_Spot": "Apply fungicide, remove infected leaves, practice crop rotation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No cure. Remove infected plants, control whitefly population.",
    "Tomato___Tomato_mosaic_virus": "No cure. Remove infected plants, disinfect tools regularly.",
    "Tomato___healthy": "No treatment needed. Keep monitoring regularly."
}

with open("model/class_names.json","r") as f:
    class_names = json.load(f)


model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, len(class_names))

model.load_state_dict(torch.load("model/plant_disease_model_v2.pth",map_location="cpu"))
model.eval()


#creating transforms so we can convert the image into tensors(numbers) and little modify the images
transform  = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import math
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Entropy calculation
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
    max_entropy = math.log(len(class_names))
    normalized_entropy = entropy / max_entropy

    # Top 2 predictions
    top2_probs, top2_indices = torch.topk(probabilities, 2)
    top1_conf = top2_probs[0][0].item() * 100
    top2_conf = top2_probs[0][1].item() * 100
    confidence_gap = top1_conf - top2_conf

    # DEBUG — ab yeh sahi jagah hai
    print(f"Top1 Conf: {top1_conf:.2f}%")
    print(f"Top2 Conf: {top2_conf:.2f}%")
    print(f"Confidence Gap: {confidence_gap:.2f}")
    print(f"Normalized Entropy: {normalized_entropy:.4f}")

    # OOD Detection
    if top1_conf < 85 or confidence_gap < 30 or normalized_entropy > 0.4:
        return {
            "disease": "Unable to detect",
            "confidence": f"{top1_conf:.2f}%",
            "treatment": "Please upload a clearer, close-up leaf image on a plain background.",
            "warning": "Low confidence — this may not be a plant leaf."
        }

    predicted_idx = top2_indices[0][0].item()
    disease = class_names[predicted_idx]
    treatment = treatments[disease]

    return {
        "disease": disease,
        "confidence": f"{top1_conf:.2f}%",
        "treatment": treatment
    }



# importing chatbot
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

from pydantic import BaseModel
class ChatRequest(BaseModel):
    disease: str
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a plant disease expert. The user has a plant infected with {request.disease}. Give short, practical advice. Answer in 2-3 sentences only."
                # "content": f"If the user asks anything unrelated to plant diseases or agriculture, politely refuse and say you can only answer plant disease related questions."
            },
            {
                "role": "user",
                "content": request.question
            }
        ]
    )
    return {"answer": response.choices[0].message.content}
   