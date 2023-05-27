import os 
import json
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

def select_white_region(mask):
  """
  From a mask, returns the middle point of the leg and the width of the leg 
  at each row. 
  """
  width = mask.shape[1] #y size of image
  mid_points = []
  leg_width = []
  for i, row in enumerate(mask): #select only the points where the mask is white
      first_pos = np.argmax(row)
      last_pos = width - np.argmax(row[::-1]) - 1

      if first_pos != 0:
        mid_points.append([int((last_pos - first_pos)//2) + first_pos, i]) #Select border points
        leg_width.append((last_pos - first_pos)) #get distance between points
        
  return mid_points, leg_width

def get_ankle_knee_pos(leg_widths, top_cut = 0.1, knee_cut = 0.3, ankle_cut = 0.3):
  """
  Returns a range of the of positions the top part and a range for 
  the lower part of the leg.
  """
  #define the limits in which to search for the minimums

  total_height_leg = len(leg_widths) 
  min_knee_y = int(total_height_leg*top_cut) #from which point to start looking for the minimum (top part of the leg)
  max_knee_y = int(total_height_leg*knee_cut) #until which point to look for the minimum (top part of the leg)
  min_ankle_y = int(total_height_leg*(1 - ankle_cut)) #from which point to start looking for the minimum (bottom part of the leg)
  max_ankle_y = int(total_height_leg*( 1 - top_cut)) #until which point to look for the minimum (bottom part of the leg)

  leg_widths = np.array(leg_widths)
  knee = np.argmin(leg_widths[min_knee_y:max_knee_y]) 
  ankle =  np.argmin(leg_widths[min_ankle_y:max_ankle_y])

  upper_part_range = slice(knee + min_knee_y, ankle + min_ankle_y)
  lower_part_range = slice(ankle + min_ankle_y, None)

  return upper_part_range, lower_part_range

def compute_pca(points):
  """
  From a 2D array computes PCA to 1 dimensions and returns repojected coordinates (line)
  """
  from sklearn.decomposition import PCA
  pca_h = PCA(1) # reduce to 1 principal components.
  converted_data = pca_h.fit_transform(points)
  return pca_h.inverse_transform(converted_data) #reproject into original coordinates 

def math_angle(s1, s2): 
  import math
  return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def slope(x1, y1, x2, y2): # Line slope given two points:
  return (y2-y1)/(x2-x1)
  
def compute_angle(category, proj_high, proj_low):
  #computing the angle https://www.cuemath.com/geometry/angle-between-two-lines/
  """
  Given a category and the two lines, it returns the exterior angle of the two lines. 
  """
  slope1 = slope(proj_low[0][0], proj_low[0][-1], proj_low[1][0], proj_low[1][-1])
  slope2 = slope(proj_high[0][0], proj_high[0][-1], proj_high[1][0], proj_high[1][-1])

  ang = math_angle(slope1, slope2)

  if category == 1: #left leg, done to compute always the exterior angle
    ang += 180 
  else:
    ang = 180 - ang

  return ang

def compute_lines_angle(mask_dict):
  """
  Given a mask dict containig the category and the mask,
  returns the lines between the floor and the ankle, the ankle and the knee
  and the exterior angle they form. 
  """
  category = mask_dict['cat']
  mask = np.array(mask_dict['mask'])

  mid_points, leg_widths = select_white_region(mask) #get the white region
  upper_part_range, lower_part_range = get_ankle_knee_pos(leg_widths)#get the two regions (up and down)

  mid_points_upper = np.array(mid_points[upper_part_range]) #select the mid points
  mid_points_lower = np.array(mid_points[lower_part_range])

  pca_upper = compute_pca(mid_points_upper) #PCA 
  pca_lower = compute_pca(mid_points_lower)

  angle = compute_angle(category, pca_upper, pca_lower) #angle

  return pca_upper, pca_lower, angle

def plot_ann_leg(mask, high, low, angle, id = None, cat = None, ax = None, alpha = 1):
  if ax is None:
    ax = plt.gca()
  ax.imshow(mask, cmap='gray', alpha = alpha)
  ax.plot(high[:,0],high[:,1], color="blue" )
  ax.plot(low[:,0], low[:,1] , color="red" )
  str_cat = ""
  str_id = ""
  if cat is not None:
    str_cat = "{}: ".format(cat)
  if id is not None:
    str_id = "{}_".format(id)  
  ax.set_title("{}{}{:.2f}°".format(str_id, str_cat, angle))
  ax.axis("off")
  return ax

def plot_lefts_rights(masks):
  
  masks_left = [mask for mask in masks if mask['cat'] == 1]
  masks_right = [mask for mask in masks if mask['cat'] == 2]

  num_plots = max(len(masks_left), len(masks_right))
  fig,ax = plt.subplots(2,num_plots,figsize=(num_plots * 3,6))

  for i, mask_dict in enumerate(masks_left):
    ax[0][i] = plot_ann_leg(mask_dict['mask'], mask_dict['highs'], mask_dict['lows'], mask_dict['angle']
                            , cat= "LEFT", ax = ax[0][i]) 

  for i, mask_dict in enumerate(masks_right):
    ax[1][i] = plot_ann_leg(mask_dict['mask'], mask_dict['highs'], mask_dict['lows'], mask_dict['angle']
                            , cat= "RIGHT", ax = ax[1][i]) 
    
  plt.show()
  
def aggregate_angles(masks, agg_function = np.mean):
  left_angles = [mask['angle'] for mask in masks if (mask['angle']>160 and mask['angle']< 190 and mask['cat'] == 1)]
  right_angles = [mask['angle'] for mask in masks if (mask['angle']>160 and mask['angle']< 190 and mask['cat'] == 2)]
  return agg_function(left_angles), agg_function(right_angles)
  
def predict(angle, overpronator = 168, supinator = 176):
  if angle < overpronator:
    return 'Overpronator'
  if angle > supinator:
    return 'Supinator'
  return 'Neutral'

def save_results(masks, input_dir, output_dir):
  masks_left = [mask for mask in masks if mask['cat'] == 1][0:3]
  masks_right = [mask for mask in masks if mask['cat'] == 2][0:3]
  
  plt.clf()
  fig,ax = plt.subplots(1,3,figsize=(3 * 3,6))
  for i, leg in enumerate(masks_left):
    img = plt.imread(os.path.join(input_dir, leg['image_path'] + '.jpg'))
    ax[i].imshow(img)
    ax[i] = plot_ann_leg(leg['mask'], leg['highs'], leg['lows'], leg['angle'],
                              cat="LEFT" , ax = ax[i], alpha = 0.5) 
  plt.savefig(os.path.join(output_dir,'left.jpg'))

  plt.clf()
  fig,ax = plt.subplots(1,3,figsize=( 3 * 3,6))
  for i, leg in enumerate(masks_right):
    img = plt.imread(os.path.join(input_dir, leg['image_path'] + '.jpg'))
    ax[i].imshow(img)
    ax[i] = plot_ann_leg(leg['mask'], leg['highs'], leg['lows'], leg['angle'],
                             cat="RIGHT" , ax = ax[i], alpha = 0.5) 
  plt.savefig(os.path.join(output_dir,'right.jpg'))