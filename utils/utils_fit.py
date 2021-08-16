import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.eval_metrics import evaluate


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, targets, net, optimizer, triplet_loss):
        with tf.GradientTape() as tape:
            # 计算loss
            outputs     = net(imgs, training=True)
            loss_value  = tf.reduce_mean(tf.losses.categorical_crossentropy(targets, outputs[0])) + triplet_loss(None, outputs[1])
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def fit_one_epoch(net, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, triplet_loss, test_loader):
    train_step  = get_train_step_fn()
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], tf.convert_to_tensor(batch[1])
            loss_value      = train_step(images, targets, net, optimizer, triplet_loss)
            loss            = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], tf.convert_to_tensor(batch[1])
            outputs         = net(images)
            loss_value      = tf.reduce_mean(tf.losses.categorical_crossentropy(targets, outputs[0])) + triplet_loss(None, outputs[1])
            val_loss        = val_loss + loss_value
            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    print("正在进行LFW数据集测试")
    labels, distances = [], []
    for _, (data_a, data_p, label) in enumerate(test_loader.generate()):
        out_a, out_p    = net(data_a)[1], net(data_p)[1]
        dists           = np.linalg.norm(out_a - out_p, axis=1)
        distances.append(dists)
        labels.append(label)
    labels      = np.array([sublabel for label in labels for sublabel in label])
    distances   = np.array([subdist for dist in distances for subdist in dist])
    _, _, accuracy, _, _, _, _ = evaluate(distances,labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % ((epoch + 1), loss / (epoch_step + 1) ,val_loss / (epoch_step_val + 1)))
