import argparse

import cv2
import random
import glob
import os
import shutil
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from cfg import *

def set_seed(seed):
    if seed != -1:
        print(seed)
        random.seed(seed) # random
        np.random.seed(seed) # np
        os.environ["PYTHONHASHSEED"] = str(seed) # os
        tf.random.set_seed(seed) # tensorflow

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH,IMAGE_HEIGHT))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1) #grayscale의 경우 추가 아니면 그냥 삭제
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) 
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.int32)
    return x

def tf_dataset(x, y, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes, dtype=tf.int32)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    mask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, num_classes])

    return image, mask


def train(args):
    data_dir_image = os.path.join(args.data_path, 'img')
    data_dir_mask = os.path.join(args.data_path, 'mask')

   
    train_x = sorted(glob.glob(os.path.join(data_dir_image, '*')))
    train_y = sorted(glob.glob(os.path.join(data_dir_mask, '*')))

    train_x, valid_x = train_test_split(train_x, test_size=args.valid_ratio)
    train_y, valid_y = train_test_split(train_y, test_size=args.valid_ratio)


    print(f"Dataset: Train: {len(train_x)} - Valid: {len(valid_x)} ")

    train_dataset = tf_dataset(train_x, train_y, batch_size=args.batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size=args.batch_size)

    NUM_TRAIN = len(train_x)

    NUM_VALID = len(valid_x)

    EPOCH_STEP_TRAIN = NUM_TRAIN // args.batch_size
    EPOCH_STEP_VALID = NUM_VALID // args.batch_size

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    print(model.name,"모델이 컴파일 되었습니다.")

    #학습부분
    print(f'Train Data:{NUM_TRAIN}')
    print(f'Valid Data:{NUM_VALID}')
    print(f'Epochs:{args.epochs}')
    print(f'Batch Size:{args.batch_size}')
    print(f'U-NET Level:{unet_level}')
    print(f'Model Name:{model.name}')
    print(f'Loss:{model.loss}')
    print(f'I/O Image Size (H * W):{IMAGE_HEIGHT} * {IMAGE_WIDTH}')

    cp_dir = os.path.join('./experiments',model.name+'.h5')

    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_dir, 
        save_best_only=True,
        verbose=1
        )

    history = model.fit(
        train_dataset,
        steps_per_epoch=EPOCH_STEP_TRAIN,
        validation_data=valid_dataset,
        validation_steps=EPOCH_STEP_VALID,
        epochs=args.epochs,
        callbacks=[mc]
        )
    
    save_dir = f'./experiments/model/{model.name}-H{IMAGE_HEIGHT}-W{IMAGE_WIDTH}-{args.batch_size}'

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    new_model_num = 1
    while (os.path.exists(f'{save_dir}/Model{new_model_num}.h5')):
        new_model_num += 1

    model.save(f'{save_dir}/Model{new_model_num}.h5')
    if os.path.exists(cp_dir):
        shutil.move(cp_dir,f'{save_dir}/Best{new_model_num}.h5')

    hist_df = pd.DataFrame(history.history) 
    with open(f'{save_dir}/History{new_model_num}.csv', mode='w') as f:
        hist_df.to_csv(f)


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default="-1")
    parser.add_argument("--epochs", type=int, default="50")
    parser.add_argument("--batch_size", type=int, default="32")
    parser.add_argument("--valid_ratio", type=float, default="0.2")
    parser.add_argument("--data_path", type=str, default="./data")

    args = parser.parse_args()

    set_seed(args.seed)
    train(args)
    

