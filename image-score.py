import cv2
import os
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import sys


def sortFunc(e):
    return e["score"]

# Function to calculate SSIM
def calculate_ssim(imageA, imageB):
    # Convert images to grayscale
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayB = cv2.resize(grayB, (780, 540), interpolation=cv2.INTER_LINEAR)
    
    # Compute SSIM between the two images
    score, _ = ssim(imageA, grayB, full=True)
    return score

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
        image = cv2.imread(image)    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (780, 540), interpolation=cv2.INTER_LINEAR)

        imageScore = []


        for (roots , dires , file) in os.walk(dire):
            for f in file:
                if '.jpg' in f or '.png' in f:

                    imageScore.append({"score":calculate_ssim(image,cv2.imread(roots+'/'+f)),"file":roots+'/'+f})
        
        imageScore.sort(key = sortFunc,reverse=True)            
        
        rows = len(imageScore)//4
        fig = plt.figure(figsize=(10, 7))
        if rows > 5:
            rows = 5
        for i in range(0,rows):
            print(i)
            for j in range(4):
                plt.subplot(rows, 4, i*4+j+1)  
                plt.imshow(cv2.cvtColor(cv2.imread(imageScore[i*4+j]['file']), cv2.COLOR_BGR2RGB))  
                plt.axis('off')  
                plt.title(f"score: {imageScore[i+j]['score']:.3f}")

        plt.show()
    
        
                                     
    
    
    
