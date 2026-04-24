import cv2
from matplotlib import pyplot as plt
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last classification layer

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path)
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

def cosine_similarity(featureA, featureB):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(featureA, featureB)


def sortFunc(e):
    return e["score"]


if __name__=="__main__":


    if len(sys.argv) != 3:
        print('no arguments or too many arguments are given')
        exit(1)
    else:
        image = sys.argv[1]
        dire = sys.argv[2]
        
    if not(os.path.isfile(image)):
        print(f'{image} does not exists')
    elif '.png' not in image and '.jpg' not in image:
        print(f'{image} is not an image file')
    elif not os.path.isdir(dire):
        print(f"{dire} is not a directory")
    else:
        imageA = preprocess_image(image)

        imageScore = []


        for (roots , dires , file) in os.walk(dire):
            for f in file:
                if '.jpg' in f or '.png' in f:
                    imageB = preprocess_image(roots+'/'+f)
                    
                    # Extract features
                    with torch.no_grad():
                        featuresA = model(imageA)
                        featuresB = model(imageB)
                    
                    # Compare using cosine similarity
                    similarity = cosine_similarity(featuresA, featuresB)
                    
                    imageScore.append({"score": similarity,"file":roots+'/'+f,'image':imageB})
                        
         
        imageScore.sort(key = sortFunc,reverse=True)            
        
        fig = plt.figure(figsize=(20, 7))
        for i in range(0,1):
            for j in range(6):
                try:
                    plt.subplot(1, 6, i*4+j+1)  
                    plt.imshow(cv2.cvtColor(cv2.imread(imageScore[i*4+j]['file']), cv2.COLOR_BGR2RGB))  
                    plt.axis('off')  
                    plt.title(f"score: {imageScore[i*4+j]['score'].item():.4f}")
                except:
                    break

        plt.show()
    
        
                                     
    
    
    
