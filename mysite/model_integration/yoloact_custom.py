import os 
import cv2
import warnings

FRAMES_DIR = "./media/videoFrames"
RESULTS_DIR = "./media/Result"

def convert_to_images(filename):
    filename = '.' + str(filename)[:-1]
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*250))    # added this line 
        cv2.imwrite( os.path.join(FRAMES_DIR, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def eval_yoloact(filename):
    print("Converting video to images")
    convert_to_images(filename)
    if len(os.listdir(FRAMES_DIR)) > 0:
        print("Generating predictions")
        images_tag = "{}:{}".format(FRAMES_DIR, RESULTS_DIR)
        weights_path = 'yolact/weights/yolact_plus_resnet50_foot_pron_201_15090_interrupt.pth'
        command = "python yolact/eval.py --trained_model {} --config yolact_resnet50_foot_pron_config --score_threshold 0.15 --top_k 1 --cuda False --output_coco_json --images {}".format(weights_path, images_tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.system(command)
