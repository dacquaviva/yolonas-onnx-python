from super_gradients.training import models
from super_gradients.common.object_names import Models

# pick one of the following models, depending on the one you trained:
# 'yolo_nas_s',
# 'yolo_nas_m',
# 'yolo_nas_l'
model = models.get("#", num_classes="#", checkpoint_path="#")
model.export("model.onnx", preprocessing=True)