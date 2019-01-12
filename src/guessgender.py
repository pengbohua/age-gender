from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128


def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None



def classify_one_multi_crop(sess, init,label_list, softmax_output, coder, images, image_file, writer):
    try:

        print('Running file %s' % image_file)
        image_batch = make_multi_crop_batch(image_file, coder)
        sess.run(init)
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
        #求和取平均
        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        #optional create a csv to save result
        if writer is not None:
            writer.writerow((image_file, best_choice[0], '%.8f' % best_choice[1]))

    except Exception as e:
        print(e)
        print('Failed to run image %s ' % image_file)

    return best_choice[0]

def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]


def guess(model_dir='./21936', class_type='genda', model_type='inception',filename='', device_id='/cpu:0', requested_step='', target='future_genda_prediction', checkpoint='14999',  face_detection_model='', face_detection_type='cascade',count=''):

    files = []
    
    if face_detection_model:
        print('Using face detector (%s) %s' % (face_detection_type, face_detection_model))
        face_detect = face_detection_model(face_detection_type, face_detection_model)
        face_files, rectangles = face_detect.run(filename)
        print(face_files)
        files += face_files

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        if class_type == 'genda':
            label_list = GENDER_LIST
        elif class_type == 'age':
            label_list = AGE_LIST
        nlabels = len(label_list)

        model_fn = select_model(model_type)

        with tf.device(device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False,count)
            init = tf.global_variables_initializer()
            
            requested_step = requested_step if requested_step else None
        
            checkpoint_path = '%s' % (model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, checkpoint)
            #加载tf模型的方法
            if count==0:
                startT=time.time()
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                endT = time.time()
                loadms = (endT - startT) * 1000
                print(loadms)

            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                if (os.path.isdir(filename)):
                    for relpath in os.listdir(filename):
                        abspath = os.path.join(filename, relpath)
                        
                        if os.path.isfile(abspath) and any([abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                            print(abspath)
                            files.append(abspath)
                else:
                    files.append(filename)
                    # If it happens to be a list file, read the list and clobber the files
                    if any([filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                        files = list_images(filename)
                
            writer = None
            output = None
            if target:
                print('Creating output file %s' % target)
                output = open(target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))
            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            print(image_files)
            for image_file in image_files:     #需要将files中的多个结果综合一下
                classify_one_multi_crop(sess, init,label_list, softmax_output, coder, images, image_file, writer)

            if output is not None:
                output.close()


def guessage(model_dir='./22801', class_type='age', model_type='inception', filename='', device_id='/cpu:0',
          requested_step='', target='future_ageprediction', checkpoint='14999', face_detection_model='',
          face_detection_type='cascade', count=''):
    files = []

    if face_detection_model:
        print('Using face detector (%s) %s' % (face_detection_type, face_detection_model))
        face_detect = face_detection_model(face_detection_type, face_detection_model)
        face_files, rectangles = face_detect.run(filename)
        print(face_files)
        files += face_files

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        if class_type == 'genda':
            label_list = GENDER_LIST
        elif class_type == 'age':
            label_list = AGE_LIST
        nlabels = len(label_list)

        model_fn = select_model(model_type)

        with tf.device(device_id):

            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False, count)
            init = tf.global_variables_initializer()

            requested_step = requested_step if requested_step else None

            checkpoint_path = '%s' % (model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, checkpoint)
            # 加载tf模型的方法
            if count == 0:
                startT = time.time()
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                endT = time.time()
                loadms = (endT - startT) * 1000
                print(loadms)

            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()

            # Support a batch mode if no face detection model
            if len(files) == 0:
                if (os.path.isdir(filename)):
                    for relpath in os.listdir(filename):
                        abspath = os.path.join(filename, relpath)

                        if os.path.isfile(abspath) and any(
                                [abspath.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                            print(abspath)
                            files.append(abspath)
                else:
                    files.append(filename)
                    # If it happens to be a list file, read the list and clobber the files
                    if any([filename.endswith('.' + ty) for ty in ('csv', 'tsv', 'txt')]):
                        files = list_images(filename)

            writer = None
            output = None
            if target:
                print('Creating output file %s' % target)
                output = open(target, 'w')
                writer = csv.writer(output)
                writer.writerow(('file', 'label', 'score'))
            image_files = list(filter(lambda x: x is not None, [resolve_file(f) for f in files]))
            print(image_files)
            for image_file in image_files:  # 需要将files中的多个结果综合一下
                classify_one_multi_crop(sess, init, label_list, softmax_output, coder, images, image_file, writer)

            if output is not None:
                output.close()
for i in range(2):
    ST=time.time()
    guessage(filename='test',count=i)
    ENDT=time.time()
    print(ENDT-ST)

#
# for i in range(10):

