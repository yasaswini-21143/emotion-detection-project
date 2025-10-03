!pip install fer opencv-python-headless

from fer import FER
import cv2
from google.colab import files
import matplotlib.pyplot as plt

# Upload an image
print("Upload a photo with a face:")
uploaded = files.upload()

# Get the filename
filename = list(uploaded.keys())[0]

# Load and analyze
img = cv2.imread(filename)
detector = FER()
result = detector.detect_emotions(img)

# Show results
print("\nEMOTION DETECTION RESULTS:")
if result:
    emotions = result[0]['emotions']
    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {score*100:.1f}%")
        
# Display image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
