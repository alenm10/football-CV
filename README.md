# football-homography

# Dataset
Dataset consist of images (from 2014 World Cup) labeled with 29 selected keypoints.
Dataset was split in 80:10:10 ratio (train (449 images) / validation (58 images) / test (58 images))

|File name|Description|Download|
|---------|---------|----|
|data_train_val_test_final.zip|Contains data for training segmentation model|[link](https://drive.google.com/file/d/1yIpuVm8i6GQjw4AisogOub7lmmU5NGcX/view?usp=sharing)|
|tracking_dataset_final.zip|Containts data for training player detection model|[link](https://drive.google.com/file/d/1RqKGY-ksyZJWMGff3fHhFrlASwCF14_x/view?usp=sharing)|

# Pipeline
![model](https://user-images.githubusercontent.com/42214173/132946696-4144b812-af44-415f-ab2c-cc6859661aa1.png)


## 1.Keypoints detection
![image](https://user-images.githubusercontent.com/42214173/132948015-a3dd09b0-eea6-4e03-946d-30dc7bcb207d.png)

First task is to detect keypoints in a given image. The model is based on an pretrained encoder backbone with FPN architecture for predicting the mask of each visible keypoint. Model was trained for 100 epochs, batch size of 4 due to memory limits, learning rate of 0.0001, sum of dice and focal loss as loss functions and applied augmentations during training (horizontal flip, random contrast, random brightness and motion blur).

Input to the model is 320x320x3 image and the output is predicted masks for detected keypoints with dimensions of 320x320x30 (29 classes for each keypoint and one for background).

![image](https://user-images.githubusercontent.com/42214173/132947725-7fb8361d-3184-4cf9-be0b-4407e1eed702.png)
<p align="center">Left: Input image (320x320x3), Right: detected keypoints mask (320x320x30)</p>

## 2.Homography estimation

Knowing the classes and coordinates of each predicted keypoint, we can compute source and destination points from the image and the 2D view of the target image.

After calculating source and destination points, we can apply OpenCV library and estimate homography matrix. Output of this step is 3x3 homography matrix.

With estimated homography, we can warp target image to the input image and make a comparison as shown in the image below.

![image](https://user-images.githubusercontent.com/42214173/132947798-ae5c9ecd-a9ae-48f2-b943-b9f73c0794c2.png)

## 3. Player detection:

YOLOv5, fast and real-time single stage object detector, is used for player detection. It takes an image as an input and output detected objects of class human with bounding box coordinates. YOLO is chosen for its speed, since the end goal is to apply this method to video, process each frame and extract players coordinates.

Model was trained on 449 training images to recognize class person for 50 epoch and batch size of 16. Since we have homography matrix from previous step, we can apply the matrix to transform player coordinates to 2D space.

![image](https://user-images.githubusercontent.com/42214173/132947803-2969225d-cbc7-481f-8af5-3ce09c484a77.png)

## 4. Player clustering

To assign color to each player we first need to extract each player region by taking bounding box coordinates.
![image](https://user-images.githubusercontent.com/42214173/132947805-c355f357-dcff-4d9b-8b7b-c584baa49c5b.png)
Applying k-means clustering on each region with 2 as number of clusters we can get two colors that represent each bounding box.
![image](https://user-images.githubusercontent.com/42214173/132947814-e811159d-0dbe-4ffe-813c-9283baceeb44.png)
Since we can assume that the most dominant color in each region will be green color for background (grass), we take smaller cluster to represent color of the region.

After assigning color to each player region, k-means with two clusters is used to divide players based on its colors in two groups (teams).

![image](https://user-images.githubusercontent.com/42214173/132947828-67fb8860-166e-4aa2-ba29-4f0aa1a7a3f1.png)

# Evaluation metrics
## Segmentation model
Intersection over Union (IoU) score is used to evaluate the segmentation model for predicting keypoints. The IoU is calculated between the predicted mask and the ground truth mask. 
FPN model with efficientnet backbone achieved the highest IoU on the test set - <b>0.87</b>.

## Player detection model
YOLO model was evaluated using mean average precision (mAP) metric, which is the most common metric for object detection and recognition. The model achieved <b>0.9886</b> mAP on test images with average IoU value between predicted and ground truth box of <b>0.84</b>.

# Examples
![image](https://user-images.githubusercontent.com/42214173/132948070-4c0aa90a-454b-446b-a960-1ef2e8f9acc3.png)
![image](https://user-images.githubusercontent.com/42214173/132948072-597cea8a-3e87-455c-918f-2f0582c78923.png)
![image](https://user-images.githubusercontent.com/42214173/132948074-9ae50ac8-3a21-4555-bab1-e8f4cadd51e9.png)

# Trained models
|File name|Description|Download|
|---------|---------|----|
|yolotrain1024.zip|Contains trained player detection model (0.98 mAP)|[link](https://drive.google.com/file/d/12NslLN8Qvz8wG0kDPGKfn6FwMZDuLeJZ/view?usp=sharing)|
|FPN-mobilenet|FPN model with mobilenet backbone (0.87 IoU)|[link](https://drive.google.com/file/d/1ThBu25TCERNx_zUQ4KOSv-9azedPVmuu/view?usp=sharing)|
|FPN-efficientnet|FPN model with efficientnet backbone (0.87 IoU)|[link](https://drive.google.com/file/d/11X6MZgO681BsnNQxOkEBTDM7ogB_iku8/view?usp=sharing)|
|U-Net-resnet|U-Net model with resnet backbone (0.83 IoU)|[link](https://drive.google.com/file/d/1ANQ1W-F7AOf75olbhYe7NFXE1W3QoAMH/view?usp=sharing)|
|U-Net-mobilenet|U-Net model with mobilenet backbone (0.84 IoU)|[link](https://drive.google.com/file/d/1ANQ1W-F7AOf75olbhYe7NFXE1W3QoAMH/view?usp=sharing)|
|U-Net-efficientnet|U-Net model with efficientnet backbone (0.82 IoU)|[link](https://drive.google.com/file/d/1Mw2lVxPl-CRcwh73fgVKfL6nbc8cUTl0/view?usp=sharing)|
