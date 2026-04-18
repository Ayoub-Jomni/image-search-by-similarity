import cv2

from skimage.metrics import structural_similarity as ssim

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


# Load sample images
imageA = cv2.imread('/home/aybi/SI/indexation/projet/image-search-by-similarity/imageA.jpg')
imageB = cv2.imread('/home/aybi/SI/indexation/projet/image-search-by-similarity/imageB.jpg')

# Calculate SSIM
ssim_score = calculate_ssim(imageA, imageB)
print(f"SSIM score: {ssim_score}")
