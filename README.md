# Instance Segmentation using Detectron2

## Overview

This repository contains a Detectron2 model trained for the task of instance segmentation on images with mask and no mask. The model is built using the Detectron2 library, a powerful and flexible computer vision library based on PyTorch.

## Model Overview

- **Task:** Instance Segmentation
- **Classes:** Mask, No Mask
- **Architecture:** Mask R-CNN with a backbone architecture (e.g., R50-FPN)
- **Model Zoo Link:** [Link to Model Zoo](https://detectron2.modelzoo.ml/detectron2/your-model-name)

## Image Labelling and Pre-processing

To label images I have used labelme,
To install `labelme`, you can use the following command:

```bash
pip install labelme
```

This command installs the `labelme` package and its dependencies. After installation, you should be able to use `labelme` for annotating images and creating datasets for computer vision tasks.

After Labelling the images the next step is to prepare the COCO format json file of our dataset (train,test and valid).

You can refer to and run the 'labelme2coco.py' file for this task in the terminal in the following way:

```bash
python labelme2coco.py test_images_folder_name --output coco_test_output.json
```

Perform this for all train, test and valid images folder.

## Note: Please check for placeholders in train_detectron2_model.py file in case.

## Getting Started

### Installation

To run the model, you'll need to install the required dependencies:

```bash
pip install -U torch torchvision
!python -m pip install pyyaml==5.1
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Model Zoo

For convenient use, the model is available in the Detectron2 Model Zoo. You can download the pre-trained model weights from [Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).

## Usage

### Inference

To use the model for inference on your own images, you can follow the example code below:

note: This code is available in the model.ipynb file as well

```python
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Load the model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("path/to/your/model_config.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("path/to/your/model_config.yaml")
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust this value based on available GPU memory
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
#cfg.DATASETS.TEST = ("test_dataset", )
#predictor = DefaultPredictor(cfg)
# Load model configuration (modify as needed)
#cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
#model.load_state_dict(torch.load("/content/drive/MyDrive/facemask_project/detectron2_maskdetectionv2.pth", map_location=torch.device("cpu")))
cfg.MODEL.WEIGHTS = "path/to/your/model.pth"
model = build_model(cfg)
model.eval()

image_path = "path/to/your/image.jpg"  # Replace with the path to your input image
img = Image.open(image_path)
img_array = np.array(img)

inputs = {"image": torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1), "height": img_array.shape[0], "width": img_array.shape[1]}

# Perform inference
with torch.no_grad():
    predictions = model([inputs])

# Post-process the predictions
instances = predictions[0]["instances"].to("cpu")
boxes = instances.pred_boxes.tensor.numpy()
scores = instances.scores.numpy()
classes = instances.pred_classes.numpy()

# Print or use the results as needed
print("Predicted Boxes:", boxes)
print("Predicted Scores:", scores)
print("Predicted Classes:", classes)


img_array = np.array(img)


v = Visualizer(img_array[:, :, ::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),scale=0.8)
out = v.draw_instance_predictions(predictions[0]["instances"].to("cpu"))
plt.imshow(out.get_image()[:, :, ::-1])

```

### Model Tuning

If you want to fine-tune the model on your own dataset, you can refer to the training script in the `model` directory. Update the configuration file and run the script to fine-tune the model.

```bash
python model/train_detectron2_model.py --config-file path/to/your/config.yaml
```


## Acknowledgments

- Thanks to the Detectron2 team for providing a robust library for computer vision tasks.

Feel free to contribute, report issues, or suggest improvements!

---