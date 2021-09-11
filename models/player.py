class Player():
  def __init__(self, 
               box, 
               image, 
               show_image_regions=False, 
               show_colors_bar=False):
    '''
      Class representing a player and his team color
      Parameters:
        box (int) - [x1,y1,x2,y2] coordinates of top left and bottom right 
        image (str) - image of the pitch where player was detected
        show_image_regions (boolean) - plot image regions 
        show_colors_bar (boolean) - plot color bar
    '''
    self.show_image_regions = show_image_regions
    self.show_colors_bar = show_colors_bar
    self.box = box
    self.coordinate_2d = None
    self.image = image
    self.player_color = self.cluster_pixels()
    self.team_color = None

  def get_box_coordinates(self):
    'Return bounding box coordinates'
    return self.box
    
  def get_player_image_region(self):
    'Return image region of player on original input image'
    x_1 = self.box[0]
    y_1 = self.box[1]
    x_2 = self.box[2]
    y_2 = self.box[3]
    player_region = self.image[y_1:y_2,x_1:x_2]
    if self.show_image_regions:
      self._visualize(player = player_region)
    return player_region

  def cluster_pixels(self):
    '''
      Clusters pixels of detected player region. 
      Pick cluster with fewer elements as player final color
    '''
    player_region = self.get_player_image_region()
    player_region = player_region.reshape((player_region.shape[0] * player_region.shape[1], 3))

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(player_region)
    min_idx = np.argmin(self._centroid_histogram(kmeans))
    player_color = kmeans.cluster_centers_[min_idx]

    if self.show_colors_bar:
      hist = self._centroid_histogram(kmeans)
      bar = self._plot_colors(hist, kmeans.cluster_centers_)
      plt.figure()
      plt.axis("off")
      plt.imshow(bar)
      plt.show()
    return player_color

  def _centroid_histogram(self, kmeans):
    'Histogram of colors in players region'
    numLabels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
    (hist, _) = np.histogram(kmeans.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

  def _plot_colors(self, hist, centroids):
    'Plots color bar'
    bar = np.zeros((50, 150, 3), dtype = "uint8")
    startX = 0
    for (percent, color) in zip(hist, centroids):
      endX = startX + (percent * 150)
      cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                    color.astype("uint8").tolist(), -1)
      startX = endX
    return bar
  
  def __str__(self):
    return "player = [x1 {}] [y1 {}] [x2 {}] [y2 {}]".format(box[0], box[1], box[2], box[3])

  def _visualize(self, **images):
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