import os
import cv2  
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import backend as k
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_from_img_path(rows,columns,list_img_path,list_mask_path):
    fig = plt.figure(figsize=(12,12))
    for i in range(1,rows*columns +1):
        fig.add_subplot(rows,columns,i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask,alpha=0.4)
    plt.show()

""" After mask adjustment, if the value is <=0.5 then that mask
    will be considered a complete black one and does not have any tumor """
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(
    data_frame,batch_size,
    aug_dict,image_color_mode="rgb",
    mask_color_mode="grayscale",image_save_prefix="image",
    mask_save_prefix="mask",flag_multi_class=False,
    num_class=2,save_to_dir=None,
    target_size=(256,256),seed=1):
  

  """
    Can generate image and mask at the same time.
    Use the same seed for image_datagen and mask_datagen 
    to ensure the transformation for image and mask is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

  """
  image_datagen= ImageDataGenerator(**aug_dict)
  mask_datagen = ImageDataGenerator(**aug_dict)
  image_generator = image_datagen.flow_from_dataframe(
      data_frame,x_col="image_filenames_train",
      class_mode=None,color_mode=image_color_mode,
      target_size=target_size,batch_size=batch_size,
      save_to_dir=save_to_dir,save_prefix=image_save_prefix,
      seed=seed
  )

  mask_generator= mask_datagen.flow_from_dataframe(
      data_frame,x_col="mask",
      class_mode=None,color_mode=mask_color_mode,
      target_size=target_size,batch_size=batch_size,
      save_to_dir=save_to_dir,save_prefix=mask_save_prefix,
      seed=seed
  )
  train_generator=zip(image_generator,mask_generator)
  for (img,mask) in train_generator:
    imag,mask = adjustData(img,mask,flag_multi_class,num_class)
    yield(img,mask)
##

def dice_coefficients(y_true,y_pred,smooth=100):
  y_true_flatten = k.flatten(y_true)
  y_pred_flatten = k.flatten(y_pred)

  intersection = k.sum(y_true_flatten*y_pred_flatten)
  union = k.sum(y_true_flatten) + k.sum(y_pred_flatten)
  return (2* intersection + smooth) / (union + smooth)


def dice_coefficients_loss(y_true,y_pred,smooth=100):
  return -dice_coefficients(y_true,y_pred)

def iou(y_true,y_pred,smooth=100):
  ## we put multibly cuz we need to get one only if the mask and the prediction have 1 for each pixel
  intersection = k.sum(y_true * y_pred)
  sum = k.sum(y_true + y_pred)
  iou = (intersection+smooth)/(sum - intersection + smooth)
  return iou

def jaccard_distance(y_true,y_pred):
  y_true_flatten = k.flatten(y_true)
  y_pred_flatten = k.flatten(y_pred)
  return -iou(y_true_flatten,y_pred_flatten)