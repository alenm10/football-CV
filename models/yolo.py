class YOLO():
  '''
    Class representing trained yolo model for player
    detection from ultralytics/yolov5
  '''
  def __init__(self, 
               weights=YOLO_WEIGHTS, 
               imgsz=320,
               conf_thres=0.50,
               iou_thres=0.45,
               classes=0):
    self.weights = weights
    self.imgsz = imgsz
    self.conf_thres = conf_thres
    self.iou_thres = iou_thres
    self.classes = classes
    
  def detect_players(self, source):
    'Returns list of detected players bounding boxes'
    PLAYERS = []

    #set_logging()
    device = select_device('')

    # Load model
    w = self.weights[0] if isinstance(self.weights, list) else self.weights
    pt = w.endswith('.pt')
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
      
    model = attempt_load(self.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
      
    imgsz = check_img_size(self.imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
      if pt:
        img = torch.from_numpy(img).to(device)
        img = img.float()   # uint8 to fp16/32
      elif onnx:
        img = img.astype('float32')
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

        # Inference
        if pt:
          visualize = False
          pred = model(img, augment=False, visualize=visualize)[0]
        elif onnx:
          pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
              
          p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

          s += '%gx%g ' % img.shape[2:]  # print string
          gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
          imc = im0  # for save_crop
          if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
              n = (det[:, -1] == c).sum()  # detections per class
              s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            print("\n")
            for *xyxy, conf, cls in reversed(det):
              line = (cls, *xyxy, conf)   # label format
              cls, x1, y1, x2, y2, conf = line[0].item(), line[1].item(), \
                                          line[2].item(), line[3].item(), \
                                          line[4].item(), line[5].item()  
              if conf >= self.conf_thres:
                PLAYERS.append([x1, y1, x2, y2])
    return PLAYERS