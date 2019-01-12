# Age-genda DeepLearning Module for Asian 
## Goal

Do face detection and age & gender classification on pictures with TensorFlow for Age Estimation for Asian

This is part of an open source project called ‘AI LED Mirror’ which was an embedded system that combined face detection, age & gender prediction and voice interaction as a whole.To address the non-stationary property of aging patterns,
age estimation can be cast as an ordinal regression problem.In this paper, we propose an End-to-End learning approach to address ordinal regression problems using deep Convolutional Neural Network, which could simultaneously conduct feature learning and
regression modeling. In particular, an ordinal regression problem is transformed into a series of binary classification sub-problems. And we propose a multiple output CNN learning algorithm to collectively solve these classification sub-problems, so that the correlation between these tasks could be explored.

**The training pipeline include face images aligning, face detection and classification.**
For face detection, you can either use OpenCV's cascade object detector in detect.py or Yolodetector in yolodetect.py
###Running
There are several ways to use a pre-existing checkpoint to do age or gender classification.  By default, the code will simply assume that the image you provided has a face in it, and will run that image through a multi-pass classification using the corners and center.

  The --class_type parameter controls which task(age/gender), and the --model_dir controls which checkpoint to restore.  There are advanced parameters for the checkpoint basename (--checkpoint) and the requested step number if there are multiple checkpoints in the directory (--requested_step)

For age estimation,call: 
```
$ python guess.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/age_test_fold_is_1/run-20854 --filename ./test1.jpg
```
For genda estimation,call: 
```
$ python guessgender.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376 --requested_step 9999 --filename ./test1.jpg
```
OR
```
$ python guess.py --model_dir /home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/gen_test_fold_is_0/run-31376 --class_type gender --requested_step 9999 --filename ./test1.jpg
```    


### Face Detection

If you have an image with one or more frontal faces, you can run a face-detector upfront, and each detected face will be chipped out and run through classification individually.  A variety of face detectors are supported including OpenCV, dlib and YOLO

OpenCV:

```
python guess.py --model_type inception --model_dir /data/xdata/rude-carnie/checkpoints/age/inception/22801 --filename /home/dpressel/Downloads/portraits/p_and_d.jpg --face_detection_model /usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml
```

To use dlib, you will need to install it and grab down the model:

```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 bunzip2 shape_predictor_68_face_landmarks.dat.bz2
pip install dlib
python guess.py --model_type inception --model_dir /data/xdata/rude-carnie/checkpoints/age/inception/22801 --filename ~/Downloads/portraits/halloween15.jpg --face_detection_type dlib --face_detection_model shape_predictor_68_face_landmarks.dat
```

YOLO tiny:

```
python guess.py --model_type inception --model_dir /data/xdata/rude-carnie/checkpoints/age/inception/22801 --filename /home/dpressel/Downloads/portraits/p_and_d.jpg --face_detection_model weights/YOLO_tiny.ckpt --face_detection_type yolo_tiny
```

If you want to run YOLO, get the tiny checkpoint from here

https://github.com/gliese581gg/YOLO_tensorflow/

### Pre-trained Checkpoints
You can use our pre-trained age checkpoint for inception_v3 here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWbDZ4Y21GLWxtV1E

A pre-trained gender checkpoint for inception is available here:

https://drive.google.com/drive/folders/0B8N1oYmGLVGWemZQd3JMOEZvdGs                          
**mymain contain the interface for a combination of functions powered by Raspberry Pi3 Debian, feel free if you do not need it**
