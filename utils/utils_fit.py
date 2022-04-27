import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.utils_metrics import evaluate


#----------------------#
#   防止bug
#----------------------#
def get_train_step_fn(strategy):
    @tf.function
    def train_step(imgs, targets, net, optimizer, triplet_loss):
        with tf.GradientTape() as tape:
            #----------------------#
            #   计算loss
            #----------------------#
            outputs             = net(imgs, training=True)
            CE_loss_value       = tf.reduce_mean(tf.losses.categorical_crossentropy(targets, outputs[0]))
            triplet_loss_value  = triplet_loss(None, outputs[1])
            loss_value          = CE_loss_value + triplet_loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value, triplet_loss_value, CE_loss_value
    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_train_step(imgs, targets, net, optimizer, triplet_loss):
            per_replica_losses, per_replica_triplet_loss_value, per_replica_CE_loss_value = strategy.run(train_step, args=(imgs, targets, net, optimizer, triplet_loss))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_triplet_loss_value, axis=None), \
                strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_CE_loss_value, axis=None)
        return distributed_train_step

#----------------------#
#   防止bug
#----------------------#
def get_val_step_fn(strategy):
    @tf.function
    def val_step(imgs, targets, net, optimizer, triplet_loss):
        #----------------------#
        #   计算loss
        #----------------------#
        outputs             = net(imgs, training=False)
        CE_loss_value       = tf.reduce_mean(tf.losses.categorical_crossentropy(targets, outputs[0]))
        triplet_loss_value  = triplet_loss(None, outputs[1])
        loss_value          = CE_loss_value + triplet_loss_value
        return loss_value, triplet_loss_value, CE_loss_value
    if strategy == None:
        return val_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_val_step(imgs, targets, net, optimizer, triplet_loss):
            per_replica_losses, per_replica_triplet_loss_value, per_replica_CE_loss_value = strategy.run(val_step, args=(imgs, targets, net, optimizer, triplet_loss))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_triplet_loss_value, axis=None), \
                strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_CE_loss_value, axis=None)
        return distributed_val_step

def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, triplet_loss, test_loader, lfw_eval_flag, save_period, save_dir, strategy):
    train_step  = get_train_step_fn(strategy)
    val_step    = get_val_step_fn(strategy)

    loss                = 0
    total_triple_loss   = 0
    total_CE_loss       = 0 

    val_loss            = 0
    val_triple_loss     = 0
    val_CE_loss         = 0 
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            loss_value, triplet_loss_value, CE_loss_value = train_step(images, targets, net, optimizer, triplet_loss)
            loss                = loss + loss_value
            total_triple_loss   = total_triple_loss + triplet_loss_value
            total_CE_loss       = total_CE_loss + CE_loss_value

            pbar.set_postfix(**{'total_loss'        : float(loss) / (iteration + 1), 
                                'total_triple_loss' : float(total_triple_loss) / (iteration + 1), 
                                'total_CE_loss'     : float(total_CE_loss) / (iteration + 1), 
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets     = batch[0], batch[1]
            loss_value, triplet_loss_value, CE_loss_value = val_step(images, targets, net, optimizer, triplet_loss)

            val_loss            = val_loss + loss_value
            val_triple_loss     = val_triple_loss + triplet_loss_value
            val_CE_loss         = val_CE_loss + CE_loss_value
            
            pbar.set_postfix(**{'val_loss'          : float(val_loss) / (iteration + 1),
                                'val_triple_loss'   : float(val_triple_loss) / (iteration + 1), 
                                'val_CE_loss'       : float(val_CE_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    if lfw_eval_flag:
        print("正在进行LFW数据集测试")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            out_a, out_p    = net(data_a)[1], net(data_p)[1]
            dists           = np.linalg.norm(out_a - out_p, axis=1)
            distances.append(dists)
            labels.append(label)
        labels      = np.array([sublabel for label in labels for sublabel in label])
        distances   = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy, _, _, _, _ = evaluate(distances,labels)
        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.h5' % ((epoch + 1), loss / epoch_step ,val_loss / epoch_step_val)))
