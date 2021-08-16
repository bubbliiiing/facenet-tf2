from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

from nets.facenet import facenet
from nets.facenet_training import FacenetDataset, triplet_loss
from utils.callbacks import (ExponentDecayScheduler, LFW_callback,
                             ModelCheckpoint)
from utils.LFWdataset import LFWDataset
from utils.utils_fit import fit_one_epoch


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用eager模式训练
    #----------------------------------------------------#
    eager       = False
    #--------------------------------------#
    #   输入图像大小
    #--------------------------------------#
    # input_shape = [112,112,3]
    input_shape = [160,160,3]
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone    = "mobilenet"
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = "./cls_train.txt"
    num_classes     = get_num_classes(annotation_path)

    model           = facenet(input_shape, num_classes, backbone=backbone, mode="train")
    #------------------------------------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显
    #   网络训练的结果也不会好，数据的预训练权重对不同数据集是通用的，因为特征是通用的
    #------------------------------------------------------------------------------------#
    model_path  = "model_data/facenet_mobilenet.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint_period   = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr           = ExponentDecayScheduler(decay_rate = 0.92, verbose = 1)
    early_stopping      = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard         = TensorBoard(log_dir='logs/')

    #----------------------#
    #   LFW估计
    #----------------------#
    test_loader         = LFWDataset(dir="./lfw", pairs_path="model_data/lfw_pair.txt", batch_size=32, image_size=input_shape)
    lfw_callback        = LFW_callback(test_loader)

    #-------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    #-------------------------------------------------------#
    val_split   = 0.05
    with open(annotation_path) as f:
        lines   = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines)*val_split)
    num_train   = len(lines) - num_val

    if backbone=="mobilenet":
        freeze_layer = 81
    elif backbone=="inception_resnetv1":
        freeze_layer = 440
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))

    for i in range(freeze_layer):
        model.layers[i].trainable = False
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        Batch_size      = 64
        Lr              = 1e-3
        Init_epoch      = 0
        Freeze_epoch    = 50

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val   // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        
        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset   = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)
                
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataset.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataset.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = Batch_size).prefetch(buffer_size = Batch_size)
            gen_val = gen_val.shuffle(buffer_size = Batch_size).prefetch(buffer_size = Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = Lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(Init_epoch, Freeze_epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            Freeze_epoch, triplet_loss(batch_size=Batch_size), test_loader)
        else:
            model.compile(
                loss={
                    'Embedding' : triplet_loss(batch_size=Batch_size),
                    'Softmax'   : "categorical_crossentropy",
                }, 
                optimizer = Adam(lr=Lr),
                metrics={
                    'Softmax'   : 'accuracy'
                }
            )
            model.fit_generator(
                train_dataset,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataset,
                validation_steps    = epoch_step_val,
                epochs              = Freeze_epoch,
                initial_epoch       = Init_epoch,
                callbacks           = [checkpoint_period, reduce_lr, early_stopping, tensorboard, lfw_callback]
            )

    for i in range(freeze_layer):
        model.layers[i].trainable = True
    if True:
        Batch_size      = 32
        Lr              = 1e-4
        Freeze_epoch    = 50
        Epoch           = 100
        
        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val   // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        
        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_train, num_classes, Batch_size)
        val_dataset   = FacenetDataset(input_shape, lines[num_train:], num_val, num_classes, Batch_size)
                
        if eager:
            gen     = tf.data.Dataset.from_generator(partial(train_dataset.generate), (tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataset.generate), (tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = Batch_size).prefetch(buffer_size = Batch_size)
            gen_val = gen_val.shuffle(buffer_size = Batch_size).prefetch(buffer_size = Batch_size)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = Lr, decay_steps = epoch_step, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

            for epoch in range(Freeze_epoch, Epoch):
                fit_one_epoch(model, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            Freeze_epoch, triplet_loss(batch_size=Batch_size), test_loader)
        else:
            model.compile(
                loss={
                    'Embedding'     : triplet_loss(batch_size=Batch_size),
                    'Softmax'       : "categorical_crossentropy",
                }, 
                optimizer = Adam(lr=Lr),
                metrics={
                    'Softmax'       : 'accuracy'
                }
            )
            print('Train with batch size {}.'.format(Batch_size))
                
            model.fit_generator(
                train_dataset,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataset,
                validation_steps    = epoch_step_val,
                epochs              = Epoch,
                initial_epoch       = Freeze_epoch,
                callbacks           = [checkpoint_period, reduce_lr, early_stopping, tensorboard, lfw_callback]
            )
