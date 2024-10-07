# Chikawa Classifier

There are 3 steps in this repo

1. You need to install the video of Chikawa in `images/videos`
2. You need to extract the frame which can decrease the time spending of labeling, please see `images/extract_frame.py`
3. After that, you need to use `image_labeler.py` which has the UI to label the image.
4. Transfer the image to dataset. `CNNs/dataset`
5. Train the model, reference to `CNNs/cnn.ipynb`
6. After that, test your model, also in `CNNs/cnn.ipynb`. The test images can be found in internet `CNNs/test_images`

This repo provided the dataset.

---

### A. Extract  Frames
- remove the duplicate / high similarity frames in video
- use ssim

here is the code example

```python

from skimage.metrics import structural_similarity as ssim
import cv2

# 读取图像
imageA = cv2.imread('imageA.png', cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread('imageB.png', cv2.IMREAD_GRAYSCALE)

# 计算结构相似性
score, diff = ssim(imageA, imageB, full=True)

print(f'SSIM: {score}')

```

### B. Image Labeler
- Resize the image ratio, add the black padding; image.shape: (360, 616, 3)
- Edge Detection: Grayscale Conversion > Gaussian Blur > Adaptive Thresholding > Canny Edge Detection
- morphological operations (填補空隙 eg. -- -- to be -----)
- Calculate the object size and select top10
- Labeling in UI via  `image_label.py`

### C. Model
- Train the model 
- CNN model, CrossEntropyLoss, AdamOptimizer
- And do the classification
  