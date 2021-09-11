class Pitch():
  'Class that represents 2D pitch with players'
  def __init__(self, 
               image_path, 
               players_boxes,
               show_image_regions=False, 
               show_colors_bar=False):
    '''
      Parameters:
        image_path (str) - path to the image from the dataset
        players_boxes (int) - (x1,y1,x2,y2) coordiantes of top left and bottom right corners
        show_image_regions (boolean) - plot image regions  
        show_colors_bar (boolean) - plot color bars
    '''
    self.image_path = image_path
    self.players_boxes = players_boxes
    self.show_image_regions = show_image_regions
    self.show_colors_bar = show_colors_bar

    self.image = cv2.imread(image_path)
    self.player_objects = self._load_players()
    self.cluster_players()

  def cluster_players(self):
    'Clusters all players into two teams by their color'

    player_colors = []
    for p in self.player_objects:
      player_colors.append(list(p.player_color))

    kmean = KMeans(n_clusters = 2)
    kmean.fit(player_colors)

    for player, team_color_id in zip(self.player_objects, kmean.labels_):
      player.team_color = kmean.cluster_centers_[team_color_id]

  def show_pitch_with_players(self, with_input_image=False):
    'Displays input image with detected players in bounding box'

    image_copy = self.image.copy()
    for player in self.player_objects:
      x1, y1, x2, y2 = player.get_box_coordinates()
      image_copy = cv2.rectangle(image_copy, (x1, y1), (x2, y2), (player.team_color) , 2) 
    if with_input_image:
      self._visualize(input_image = self.image, detected_players = image_copy)
    else:
      self._visualize(detected_players = image_copy)

  def show_pitch_with_players_H(self, homography, with_input_image=False):
    'Displays 2d reprezentation of the pitch with detected players'

    for player in self.player_objects:
      c = self.get_field_coordinates(player.get_box_coordinates(), homography)
      player.coordinate_2d = c
    
    pitch_image_2d = cv2.imread("/content/drive/MyDrive/pitch_template.png")
    for player in self.player_objects:
      x1 = int(player.coordinate_2d[0])
      y1 = int(player.coordinate_2d[1])
      x2 = int(player.coordinate_2d[0]+5)
      y2 = int(player.coordinate_2d[1]+5)
      pitch_image_2d = cv2.rectangle(pitch_image_2d, (x1, y1), (x2, y2), (player.team_color), 4) 
    
    pitch_image_2d = cv2.resize(pitch_image_2d, (400,320))

    if with_input_image:
      self._visualize(input_image = self.image, top_down_view = pitch_image_2d)
    else:
      self._visualize(top_down_view = pitch_image_2d)

  def get_field_coordinates(self, player, pred_h):
    'Returns players final coordinates in 2D space after applying homography'

    x_1 = player[0]
    y_1 = player[1]
    x_2 = player[2]
    y_2 = player[3]
    x = (x_1 + x_2) / 2.0 
    y = max(y_1, y_2)
    pts = np.array([float(x), float(y)])
    dst = self._warp_point(pts, np.linalg.inv(pred_h))
    return dst

  def _warp_point(self, point, homography):
    'Returns point after applying homography'
    dst = cv2.perspectiveTransform(np.array(point).reshape(-1, 1, 2), homography)
    return dst[0][0]
    
  def _load_players(self):
    'Creates and return Player objects from player bounding boxes'

    players_objects = []
    for x1, y1, x2, y2 in self.players_boxes:
      players_objects.append(Player(box = [int(x1), int(y1), int(x2), int(y2)], 
                                    image = self.image,
                                    show_image_regions=self.show_image_regions, 
                                    show_colors_bar=self.show_image_regions))
    return players_objects

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