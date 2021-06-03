import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import glob
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from datasets import CityscapesDataset
from models import ERFNet
from evaluation import evaluate, compute_intersection_and_union_in_batch


def main(args):

    img_h, img_w = args.img_height, args.img_width
    val_batch_size = args.val_batch_size
    threshold = args.threshold

    dataset = CityscapesDataset()

    print('Creating network and loading weights...')
    network = ERFNet(dataset.num_classes)

    # Initialize network weights
    inp_test = tf.random.normal(shape=(1, img_h, img_w, 3))
    out_test = network(inp_test, is_training=False)
    print('Shape of network\'s output:', out_test.shape)

    # Load weights and images from given paths
    weights_path = os.path.join(os.getcwd(), args.weights)
    network.load_weights(weights_path)
    print('Weights from {} loaded correctly.'.format(weights_path))
    get_percision_on_validation_set(dataset, network, val_batch_size, (img_h, img_w),threshold)
    get_recall_on_validation_set(dataset, network, val_batch_size, (img_h, img_w),threshold)

    
def get_percision_on_validation_set(dataset, network, val_batch_size, image_size, threshold):
    total_tp = tf.zeros((1), tf.int64)
    total_tp_and_fp = tf.zeros((1), tf.int64)

    num_val_batches = dataset.num_val_images // val_batch_size
    for batch in range(num_val_batches):
        x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
        y_pred_logits = network(x, is_training=False)
        y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
        tp_batch, tp_and_fp_batch = get_precisition_in_batch(y_true_labels, y_pred_labels, dataset.num_classes, threshold)
        total_tp += tp_batch
        total_tp_and_fp += tp_and_fp_batch
        batchprecision = tf.divide(tp_batch, tp_and_fp_batch)
        print('Precistion from batch {} / {} is {}.'.format(batch+1, num_val_batches, batchprecision))
    total_set_precision = tf.divide(total_tp, total_tp_and_fp)
    print('Total Precistion is {} on IoU threshold: {}'.format(total_set_precision, threshold))
    return total_set_precision

def get_precisition_in_batch(y_true_labels, y_pred_labels, num_classes, threshold):
    tp_batch, tp_and_fp_batch = [], []
    batch_intersection_pred, batch_union_pred = compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes)
    batch_intersection_true, batch_union_true = compute_intersection_and_union_in_batch(y_true_labels, y_true_labels, num_classes)
    iou_per_class_true = tf.divide(batch_intersection_true,batch_union_true)
    iou_per_class_pred = tf.divide(batch_intersection_pred,batch_union_pred)

    iou_per_class_array_true = iou_per_class_true.numpy()
    iou_per_class_array_true[np.isnan(iou_per_class_array_true)] = 0
    #print("iou_per_class_true: {}".format(iou_per_class_array_true))

    iou_per_class_array_pred = iou_per_class_pred.numpy()
    iou_per_class_array_pred[np.isnan(iou_per_class_array_pred)] = 0
    #print("iou_per_class_pred: {}".format(iou_per_class_array_pred))

    pred_non_zero = iou_per_class_array_pred[iou_per_class_array_pred != 0]
    #print("pred nonzero: {}".format(pred_non_zero))

    iou_per_class_threshold = np.where(pred_non_zero > threshold, 1, 0)
    #print("iou_per_class_threshold: {}".format(iou_per_class_threshold))

    tp = len(iou_per_class_threshold[iou_per_class_threshold != 0])
    tp_and_fp = len(iou_per_class_threshold)
    tp_batch.append(tp)
    tp_and_fp_batch.append(tp_and_fp)

    return tp_batch, tp_and_fp_batch      

def get_recall_on_validation_set(dataset, network, val_batch_size, image_size, threshold):
    total_tp = tf.zeros((1), tf.int64)
    total_tp_and_fn = tf.zeros((1), tf.int64)
    num_val_batches = dataset.num_val_images // val_batch_size
    for batch in range(num_val_batches):
        x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)
        y_pred_logits = network(x, is_training=False)
        y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)
        tp_batch, tp_and_fn_batch = get_recall_in_batch(y_true_labels, y_pred_labels, dataset.num_classes,threshold)
        total_tp += tp_batch
        total_tp_and_fn += tp_and_fn_batch
        batchrecall= tf.divide(tp_batch, tp_and_fn_batch)
        print('Recall from batch {} / {} is {}.'.format(batch+1, num_val_batches, batchrecall))
    total_set_recall = tf.divide(total_tp, total_tp_and_fn)
    total_set_recall = total_set_recall.numpy()
    total_set_recall = np.unique(total_set_recall)
    print('Total Recall is {} on Iou threshold: {}'.format(total_set_recall, threshold))
    return total_set_recall

def get_recall_in_batch(y_true_labels, y_pred_labels, num_classes, threshold):
    tp_batch, tp_and_fn_batch = [], []
    batch_intersection_pred, batch_union_pred = compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes)
    batch_intersection_true, batch_union_true = compute_intersection_and_union_in_batch(y_true_labels, y_true_labels, num_classes)
    iou_per_class_true = tf.divide(batch_intersection_true,batch_union_true)
    iou_per_class_pred = tf.divide(batch_intersection_pred,batch_union_pred)

    iou_per_class_array_true = iou_per_class_true.numpy()
    iou_per_class_array_true[np.isnan(iou_per_class_array_true)] = 0
    #print("iou_per_class_true: {}".format(iou_per_class_array_true))

    iou_per_class_array_pred = iou_per_class_pred.numpy()
    iou_per_class_array_pred[np.isnan(iou_per_class_array_pred)] = 0

    pred_non_zero = iou_per_class_array_pred[iou_per_class_array_pred != 0]

    iou_per_class_threshold = np.where(pred_non_zero > threshold, 1, 0)

    pred_bool_values = np.where(iou_per_class_array_pred > 0, 1 ,0)
    #print("pred bool values: {}".format(pred_bool_values))

    tp = len(iou_per_class_threshold[iou_per_class_threshold != 0])
    tp_and_fn = tp
    fn = 0
    for index in range(len(iou_per_class_array_true)):
        if (iou_per_class_array_true[index] > pred_bool_values[index]):
            tp_and_fn += 1
            fn +=1
    tp_batch.append(tp)
    tp_and_fn_batch.append(tp_and_fn)
    return tp_batch, tp_and_fn_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--threshold', type=float, default=0.25, help='Image height after resizing')
    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    #parser.add_argument('--weights', type=str, required=True, help='Relative path of network weights')
    parser.add_argument('--weights', type=str, default="pretrained/pretrained.h5", help='Relative path of network weights')
    args = parser.parse_args()
    main(args)
   
    