#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- come up with a batching scheme that preserved order / keeps a unique ID
"""
import numpy as np
import pandas as pd
import os
import argparse
import time
import cv2

import caffe

from detectorc import Detectorc

CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # ignore windows past the boundaries
            if ((x + windowSize[0]) <= image.shape[0] and (y + windowSize[1]) <= image.shape[1]) :
                # yield the current window
                yield (x, y, x + windowSize[0], y + windowSize[1])

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    capture = cv2.VideoCapture() 

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "bvlc_googlenet.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "bvlc_googlenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="selective_search",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        # default=os.path.join(pycaffe_dir,
        #                      'caffe/imagenet/ilsvrc_2012_mean.npy'),
        default='',
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
            
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make detector.
    detector = Detectorc(args.model_def, args.pretrained_model, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)



    # open camera
    # set the frame size
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,600)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,450)
    

    if not capture.open(0):
        print "capture from camera didn't work"


    while True:
        # # Load input.
        t = time.time()
        print("Loading input...")

        # load frams from camera
        success, image = capture.read()
        print "Image w: {} h: {}" % (images.shape[1], image.shape[0])
        # generate window coordinates
        windows = sliding_window(image, 40, (100, 100))

        # Detect.
        detections = detector.detect_windows(image, windows)
        # print detections
        print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                        time.time() - t))
        
        cv2.imshow('image',image)
        c = cv2.waitKey(10)
        if( c == 27 or c == 'q' or c == 'Q' ):
            break
        

if __name__ == "__main__":
    import sys
    main(sys.argv)
