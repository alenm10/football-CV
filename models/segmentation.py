SOURCE_DESTINATION_MAP = {
  0: [2, 2],      1: [2, 65],     2: [52, 65],    3: [2, 117],
  4: [18, 117],   5: [2, 204],    6: [18, 204],   7: [2, 254],
  8: [52, 254],   9: [2, 318],    10: [160, 2],   11: [160, 160],
  12: [160, 318], 13: [318, 2],   14: [318, 65],  15: [269, 65],
  16: [318, 117], 17: [303, 117], 18: [318, 204], 19: [303, 204],
  20: [318, 254], 21: [269, 254], 22: [318, 318], 23: [52, 128],
  24: [52, 193],  25: [160, 117], 26: [160, 204], 27: [269, 128],  
	28: [269, 192],
}

class SegmentationModel():
  'Class representing UNet segmentaion model for keypoints detection'
  def __init__(self, 
               weights_path='/content/drive/MyDrive/UNETmodel/UNET_iou_0.836167_resnet34.h5',
               backbone="resnet34",
               input_shape=(320, 320),
               ):
    self.weights_path = weights_path
    self.backbone=backbone
    self.input_shape=input_shape
    self.template = cv2.resize(cv2.cvtColor(cv2.imread('/content/drive/MyDrive/pitch_template.png'), 
                                                              cv2.COLOR_BGR2RGB),(1280,720))/255.
    self.model = sm.Unet(backbone,
                        classes=30,
                        activation="softmax",
                        input_shape=(input_shape[0], input_shape[1], 3),
                        encoder_weights="imagenet")
    self.model.load_weights(self.weights_path)
    self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                        loss=sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(), 
                        metrics=[sm.metrics.IOUScore(threshold=0.5)])
    
  def predict(self, image_path, show_mask=False):
    'Predicts mask for a given image'

    image = cv2.imread(image_path)
    image_org = image
    preprocessing = sm.get_preprocessing(self.backbone) 
    image = preprocessing(image)
    predicted_mask = self.model.predict(np.array([image]))
    if show_mask:
      self.visualize(
            input_image=image_org,
            predicted_mask=predicted_mask[..., -1].squeeze())
    return predicted_mask

  def get_points_from_mask(self, mask):
    'Returns source -> destination mapping of detected keypoints'
    
    # get keypoints
    keypoints = {}
    idxs = np.argwhere(mask[0][:, :, :-1] > 0.9)
    for x, y, cls in idxs:
      if cls in keypoints.keys():
        keypoints[cls][0].append(x)
        keypoints[cls][1].append(y)
      else:
        keypoints[cls] = [[x], [y]]   
    for cls in keypoints.keys():
      keypoints[cls] = [np.mean(keypoints[cls][1]) , np.mean(keypoints[cls][0])]
    
    # get source and destination points mapping
    src_pts = []
    dst_pts = []
    for cls, xy in keypoints.items():
      # print("{} - {}".format(id_kp, v))
      src_pts.append(xy)                            # x and y for calculated mean x,y on image
      dst_pts.append(SOURCE_DESTINATION_MAP[cls])   # find that class on target image
      # list_ids.append(cls)
    src, dst = np.array(src_pts), np.array(dst_pts)
    return src, dst
  
  def __call__(self, image_path):
    'Return predicted homography for a given image'
    
    predicted_mask = self.predict(image_path=image_path, show_mask=True)
    src, dst = self.get_points_from_mask(predicted_mask)
    predicted_h, _ = cv2.findHomography(dst, src, cv2.RANSAC, 20)
    warped_perspective = cv2.warpPerspective(cv2.resize(self.template, (320,320)), 
                                    predicted_h, 
                                    dsize=(320,320))

    image = cv2.imread(image_path)
    self.visualize(input_image=image, 
                    warped_perspective=warped_perspective)
    return predicted_h

  def visualize(self, **images):
    'Plot images'
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()