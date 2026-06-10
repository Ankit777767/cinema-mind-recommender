import os
import torch
from tqdm import tqdm
from PIL import Image
import pandas as pd 
from torchvision import transforms,models
import torch.nn as nn

BASE_DIR = os.getcwd()
POSTERS_DIR = os.path.join(BASE_DIR, "data", "posters")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "visual_embeddings.parquet")

def generate_visual_embeddings():
  print("Loading Pre-trained ResNet-50 Model...")
  #weights=default will call the latest res50 trained model
  model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
  #we removed the classification head and replaced it with identity layer to get 5048 dimensional vectors as required

  model.fc=nn.Identity() # this will replace the final layer of fully connected layer with identoty layer

  model.eval()# when this is called, the train specific features like dropout layers are getting hidden

  # Image Transformations to fix standard in-out

  preprocess = transforms.Compose([
    transforms.Resize(256),#resize shortedt edge to 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(                # Standard ImageNet normalization values
          mean=[0.485, 0.456, 0.406], 
          std=[0.229, 0.224, 0.225]
      ),
  ])

 
  visual_data =[]
  valid_images = [f for f in os.listdir(POSTERS_DIR) if f.endswith('.jpg')]
  print(f"Found {len(valid_images)} posters to process.")
  # we'll use torch.no_grad, as we are not training the model, we are just inferencing, so it will save lat of resources
  with torch.no_grad():
    for filename in tqdm(valid_images,desc="Extracting Visual Features"):
      movie_id = filename.split('.')[0]
      img_path = os.path.join(POSTERS_DIR, filename)
      try:

        img = Image.open(img_path).convert('RGB')
        # Apply transformations and add a batch dimension: [1, 3(represents rgb conversion), 224, 224]
        img_tensor = preprocess(img).unsqueeze(0)# adds batch dimension at start , here 1
        # Pass image through the neural network
        features = model(img_tensor)
        feature_vector = features.squeeze().tolist()# converted back to python list
        visual_data.append({
              "id": int(movie_id), # Ensure ID is an integer to match our text data later
              "visual_embedding": feature_vector
          })
      except Exception as e:
                print(f"\nError processing {filename}: {e}")

# 6. Save to Parquet
    print(f"\nSaving visual embeddings to: {OUTPUT_FILE}")
    df = pd.DataFrame(visual_data)
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("✅ Visual embeddings generated successfully!")

if __name__ == "__main__":
    generate_visual_embeddings()


