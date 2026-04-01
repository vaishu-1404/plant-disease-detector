from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import json 
import io
from fastapi.middleware.cors import CORSMiddleware

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

model.load_state_dict(torch.load("model/plant_disease_model.pth",map_location="cpu"))
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

# Creating API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) #unsqueeze is used to add 1 dimension which is basically like one batch cuz mode expect (batch(32), channel(rgb-3), 224, 244)
    
    with torch.no_grad():
        outputs = model(image_tensor)

    _, predicted = torch.max(outputs, 1)
    predicted_idx = predicted.item()  #.item() is used to convert tensor into python number

    probabilites = torch.nn.functional.softmax(outputs, dim=1)
    confidence = probabilites[0][predicted_idx].item() * 100

    disease = class_names[predicted_idx]
    treatment = treatments[disease]

    return {

        "disease": disease,
        "confidence": f"{confidence:.2f}%",
        "treatment": treatment
    }
    pass