import os 
import cv2
import warnings
import numpy as np
import model_integration.postprocessing as pp

FRAMES_DIR = "./media/videoFrames"
RESULTS_DIR = "./media/Result"
FINAL_IMAGES_DIR = "./media/finalImages"

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
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    print("Converting video to images")
    if len(os.listdir(FRAMES_DIR)) > 0:
        print("Generating predictions")
        images_tag = "{}:{}".format(FRAMES_DIR, RESULTS_DIR)
        weights_path = 'yolact/weights/yolact_plus_resnet50_foot_pron_402_14879_interrupt.pth'
        command = "python yolact/eval.py --trained_model {} --config yolact_resnet50_foot_pron_config --score_threshold 0.15 --top_k 1 --cuda False --output_coco_json --images {}".format(weights_path, images_tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.system(command)


def postprocessing():
    left_leg_data = {}
    right_leg_data = {}

    if not os.path.exists(FINAL_IMAGES_DIR):
        os.makedirs(FINAL_IMAGES_DIR)

    left_count = 0 #load the data
    right_count = 0
    for path in os.listdir(RESULTS_DIR):
        if path.endswith('npy'):
            preds = np.load(os.path.join(RESULTS_DIR, path), allow_pickle=True).item()
            if preds['classes'].size != 0:
                if preds['classes'][0] == 0:
                    left_leg_data[left_count] = preds | {'img_path': path.split('.')[0]}
                    left_count += 1
                else:
                    right_leg_data[right_count] = preds | {'img_path': path.split('.')[0]}
                    right_count += 1   
    
    K = 5 #get the top five predictions for the left and right leg 
    left_leg = dict(sorted(left_leg_data.items(), key=lambda x: x[1]['scores'][0], reverse=True)[0:K])
    right_leg= dict(sorted(right_leg_data.items(), key=lambda x: x[1]['scores'][0], reverse=True)[0:K])

    left_leg = {k:{'mask':v['masks'][0], 'cat':1, 'image_path': v['img_path']} for k,v in left_leg.items()}
    right_leg = {k:{'mask':v['masks'][0], 'cat':2, 'image_path': v['img_path']} for k,v in right_leg.items()}
    all_legs = list(left_leg.values()) + list(right_leg.values())

    for leg in all_legs: #compute lines and angle for left leg 
        high, low, angle = pp.compute_lines_angle(leg)
        leg.update({'highs':high, 'lows':low, 'angle':angle})

    left_angle, right_angle = pp.aggregate_angles(all_legs)
    pred_left = pp.predict(left_angle)
    pred_right = pp.predict(right_angle)

    print("Pronation of left leg with mean angle {:.2f}° classified as: {}".format(left_angle, pred_left))
    print("Pronation of right leg with mean angle {:.2f}° classified as: {}".format(right_angle, pred_right))
    
    pp.save_results(all_legs,FRAMES_DIR, FINAL_IMAGES_DIR)

    return pred_left, pred_right
