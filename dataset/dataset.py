class Dataset:
  '''
    Dataset for train/test/val 
    Parameters:
      images_path (str) - images directory
      masks_path (str) - masks directory
      n_classes (int) - 29 keypoints classes and 1 background
      augmentation (func) - image segmentations function 
      preprocessing (func) - image preprocessing function
  '''
  def __init__(
        self,
        images_path,        # images directory
        masks_path,         # masks directory
        n_classes=30,       # 29 keypoints classes and 1 background
        augmentation=None,  # image segmentations 
        preprocessing=None, # image preprocessing 
    ):
    self.n_classes = n_classes
    self.images_dataset_paths = [os.path.join(images_path, x) for x in os.listdir(images_path) if os.path.splitext(x)[1] in ('.jpg')]
    self.masks_dataset_paths = [os.path.join(masks_path, x.replace("images", "annotations").replace("jpg", "txt")) for x in self.images_dataset_paths]
    self.len = len(self.images_dataset_paths)
    self.augmentation = augmentation
    self.preprocessing = preprocessing
    self.LEFT_RIGHT_FLIP = {
        0: 13, 1: 14, 2: 15, 3: 16, 4: 17, 5: 18, 6: 19, 7: 20,
        8: 21, 9: 22, 10: 10, 11: 11, 12: 12, 13: 0, 14: 1, 15: 2,
        16: 3, 17: 4, 18: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 27,
        24: 28, 25: 25, 26: 26, 27: 23, 28: 24,
    }
    
  def __getitem__(self, i):
    'Return a pair of image and mask after applying augmentations'
    
    image = cv2.cvtColor(cv2.imread(self.images_dataset_paths[i]), cv2.COLOR_BGR2RGB)
    keypoints = read_mask_file(self.masks_dataset_paths[i])
    if random.random() < 0.5:
      image = cv2.flip(image, 1)
      flipped_keypoints = {}
      for idx, v in keypoints.items():
        new_idx = self.LEFT_RIGHT_FLIP[idx]
        new_y = image.shape[1] - 1 - min(v[1], image.shape[1]-1)
        new_x = min(v[0], image.shape[0]-1)
        flipped_keypoints[new_idx] = (new_x, new_y)
      keypoints = flipped_keypoints

    _mask = get_mask(keypoints, image_shape=(image.shape[0], image.shape[1]))
    
    mask = np.zeros((image.shape[0], image.shape[1], self.n_classes))
    for i in range(len(_mask)):
      for j in range(len(_mask[0])):
        mask[i][j][int(_mask[i][j])] = 1

    if self.augmentation:
      sample = self.augmentation(image=image)
      image = sample["image"]

    if self.preprocessing:
      sample = self.preprocessing(image=image)
      image = sample['image']

    return image, mask

  def __len__(self):
    'Returns the number of elements in dataset'
    return len(self.images_dataset_paths)
    
class DataGenerator(tf.keras.utils.Sequence):
  '''
    Data geenrator for train/test/val 
    Parameters:
      dataset (Dataset) - Dataset class train/val/test
      batch_size (int) - batch size for training (4) and val/test (1)
      shuffle (boolean) - shuffle data after each epoch
  '''
  def __init__(self, dataset, batch_size=1, shuffle=False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.indexes = np.arange(dataset.len)
    self.on_epoch_end()

  def __getitem__(self, i):
    'Generate one batch of data'
    indexes = self.indexes[i * self.batch_size : (i+1) * self.batch_size]
    images = []
    masks = []
    for j in indexes:
      image, mask = self.dataset[j]
      images.append(image)
      masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

  def __len__(self):
    'Denotes the number of batches per epoch'
    return len(self.indexes) // self.batch_size

  def on_epoch_end(self):
    'Updates indexes after each epoch by shuffling them'
    if self.shuffle:
      np.random.shuffle(self.indexes)
      #self.indexes = np.random.permutation(self.indexes)
      
def get_training_augmentation():
  'Returns image augmentations'
  train_transform = [
    A.OneOf(
        [
          A.RandomBrightnessContrast(p=1), 
        ], 
          p=0.9,),
    A.OneOf(
        [
          A.Sharpen(p=1),
          A.Blur(blur_limit=3, p=1),
          A.MotionBlur(blur_limit=3, p=1),
        ],
          p=0.9,
    )
  ]
  return A.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
  'Returns preprocessing function'
  _transform = [
      A.Lambda(image=preprocessing_fn),
  ]
  return A.Compose(_transform)