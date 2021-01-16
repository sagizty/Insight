# tf2-keras-gradcam-flower
GradCAM implementation in Keras(Tensorflow2.3)

### Requirements
- python 2.7.9
- numpy 1.18.5
- scikit-image 0.18.1
- matplotlib 3.3.3
- tensorflow 2.3.0

## Usage
First, run preprocess.py to generate numpy data.
```bash
$ python preprrocess.py
```

next, run cnn.py.
```bash
$ python cnn.py
```

Then, run gradcam.py to visualize gradcam heatmaps.
```bash
$ python gradcam.py --sample-index 14 --batch-size 3
```


## Reference
### CNN
https://tensorflow.google.cn/tutorials/images/cnn?hl=zh_cn  
https://github.com/KeishiIshihara/keras-gradcam-mnist  

### GradCAM
https://github.com/insikk/Grad-CAM-tensorflow  
https://github.com/jacobgil/keras-grad-cam  

