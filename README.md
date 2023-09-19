# yolonas-onnx-python
Example of training <a href="https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md">YOLO-NAS</a> and exporting (<a href="https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models_export.md">ONNX</a>) as well as inferencing with python using onnxruntime.

## Setup

```
python -m venv /path/to/env

source /path/to/env/bin/activate # Linux/mac

\path\to\env\Script\activate # Windows
```
```
pip install super-gradients #install super-gradients package for training
```
```
pip install onnxruntime-gpu #install onnxruntime-gpu for model inference using ONNX model
```
## <a href="https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md">YOLO-NAS</a>   

### Train
Run the script `yolonas/train.py` to train a model on a custom dataset (make sure to save the dataset according to the used folder structure), replace # with the correct value.

```
ROOT 
├── dataset.yaml
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```

### Test
Run the script `yolonas/test.py` to test the trained model on an image, replace # with the correct value.

## <a href="https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models_export.md">ONNX</a>  
### Convert model to ONNX
Run the script `onnx/convert_to_onnx.py` to convert trained model to onnx, replace # with the correct value.
### Inference ONNX model using webcam
Run the script `onnx/inference_onnx_webcam.py` to do inference on camera usingf ONNX model, replace # with the correct value.
