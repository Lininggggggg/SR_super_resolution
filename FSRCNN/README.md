# FSRCNN-TensorFlow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## Prerequisites
 * Python 3.6
 * TensorFlow
 * Scipy version > 0.18
 * h5py
 * PIL

## Usage
You need to change main.py in 16 or 17 row to choose training or testing. Then, you can run: `python main.py`

Of course, you can specify epochs, learning rate, data directory, etc in the same way.

Check `main.py` for all the possible flags

Also includes script `expand_data.py` which scales and rotates all the images in the specified training set to expand it

## Result

Original baby image:
![orig](https://github.com/Lininggggggg/SR_super_solution/blob/master/FSRCNN/test_result/00002_HR.bmp?raw=true)

![orig](https://github.com/Lininggggggg/SR_super_solution/blob/master/FSRCNN/test_result/00002_HR_color.bmp?raw=true)

Bicubic interpolated image and color image:

![orig](https://raw.github.com/Lininggggggg/SR_super_solution/master/FSRCNN/test_result/00002_BIC.bmp?raw=true)

![orig](https://github.com/Lininggggggg/SR_super_solution/blob/master/FSRCNN/test_result/00002_BIC_color.bmp?raw=true)

FSRCNN image and color image:

![orig](https://github.com/Lininggggggg/SR_super_solution/blob/master/FSRCNN/test_result/00002_FSR.bmp?raw=true)

![orig](https://github.com/Lininggggggg/SR_super_solution/blob/master/FSRCNN/test_result/00002_FSR_color.bmp?raw=true)


## References

* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)
* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
* https://github.com/drakelevy/FSRCNN-Tensorflow
