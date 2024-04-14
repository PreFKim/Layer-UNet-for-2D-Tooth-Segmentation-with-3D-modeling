import tensorflow as tf
import keras.backend as K

def DiceBCELoss(y_true, y_pred,smooth=1e-5):    
    
    y_true_f= K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    BCE =  tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE
    
def IoULoss(y_true, y_pred, smooth=1e-6):

    y_pret_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    
    intersection = K.sum(y_true_f* y_pret_f)
    total = K.sum(y_true_f) + K.sum(y_pret_f)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def FocalLoss(y_true, y_pred, alpha=1.0, gamma=2.0):    
    
    y_pret_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    
    BCE = K.binary_crossentropy(y_true_f, y_pret_f)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def Focal_IoU(y_true, y_pred):
  return FocalLoss(y_true,y_pred) + IoULoss(y_true, y_pred)
