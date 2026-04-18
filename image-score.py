import cv2
import os
from skimage.metrics import structural_similarity as ssim
import sys

# Function to calculate SSIM
def calculate_ssim(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    grayA = cv2.resize(grayA, (780, 540), interpolation=cv2.INTER_LINEAR)
    grayB = cv2.resize(grayB, (780, 540), interpolation=cv2.INTER_LINEAR)
    
    # Compute SSIM between the two images
    score, _ = ssim(grayA, grayB, full=True)
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

        for (roots , dires , file) in os.walk(dire):
            print(roots)
            for f in file:
                if '.jpg' in f or '.png' in f:

                    print(f'{f} : {calculate_ssim(cv2.imread(image),cv2.imread(roots+"/"+f))}')
    
    
    
