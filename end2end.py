# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from math import *
from PIL import Image, ImageFont, ImageDraw
from tensorflow.python.client import timeline
from utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model
import keras

tf.app.flags.DEFINE_string('test_data_path', None, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './', '')
tf.app.flags.DEFINE_string('output_dir', './results/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

from nets import model
from pse import pse

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

from recognition import keys
from recognition.densenet import dense_cnn

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logger.info('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    # ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio=1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    # get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1] - 1, -1, -1):
        kernal = np.where(seg_maps[..., i] > thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh * ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time() - start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        # (y,x)
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals, timer


def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0] * 255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt2 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])): min(xdim-1, int(pt3[0]))]
    return imgOut


characters = keys.alphabetChinese[:]
characters = characters[1:] + u'卍'
nclass = len(characters)
print(nclass)


def _decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 0 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])

    return char_list


def main(argv=None):
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    t0 = time.time()
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    im_fn_list = get_images()
    for im_fn in im_fn_list:
        points_list = []
        tf.reset_default_graph()
        with tf.get_default_graph().as_default():
            input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                          trainable=False)
            seg_maps_pred = model.model(input_images, is_training=False)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                model_path = os.path.join(FLAGS.checkpoint_path,
                                          os.path.basename(ckpt_state.model_checkpoint_path))

                logger.info('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)

                im = cv2.imread(im_fn)[:, :, ::-1]
                draw_img = im[:, :, ::-1].copy()
                logger.debug('image file:{}'.format(im_fn))

                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                h, w, _ = im_resized.shape
                # options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                timer = {'net': 0, 'pse': 0}
                start = time.time()
                seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open(os.path.join(FLAGS.output_dir, os.path.basename(im_fn).split('.')[0]+'.json'), 'w') as f:
                #     f.write(chrome_trace)

                boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)
                logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(im_fn, timer['net'] * 1000, timer['pse'] * 1000))

                if boxes is not None:
                    boxes = boxes.reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                    h, w, _ = im.shape
                    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

                duration = time.time() - start_time
                logger.info('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(FLAGS.output_dir, '{}.txt'.format(
                        os.path.splitext(os.path.basename(im_fn))[0]))

                    with open(res_file, 'w') as f:
                        num = 0
                        for i in range(len(boxes)):
                            # to avoid submitting errors
                            box = boxes[i]
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                                continue

                            num += 1

                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                                box[2, 0], box[2, 1], box[3, 0], box[3, 1]))

                            yDim, xDim = im[:, :, ::-1].shape[:2]
                            if box[0, 0] > box[2, 0]: # box point1在右下角，顺时针
                                pt1 = (max(1, box[2, 0]), max(1, box[2, 1]))
                                pt2 = (box[3, 0], box[3, 1])
                                pt3 = (min(box[0, 0], xDim - 2), min(yDim - 2, box[0, 1]))
                                pt4 = (box[1, 0], box[1, 1])
                            else: # box point1在左下角， 顺时针
                                pt1 = (max(1, box[1, 0]), max(1, box[2, 1]))
                                pt2 = (box[2, 0], box[2, 1])
                                pt3 = (min(box[3, 0], xDim - 2), min(yDim - 2, box[3, 1]))
                                pt4 = (box[0, 0], box[0, 1])

                            points = [pt1, pt2, pt3, pt4]
                            points_list.append(points)

                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(255, 255, 0), thickness=2)

        tf.reset_default_graph()
        keras.backend.clear_session()
        input = Input(shape=(32, None, 1), name='the_input')
        y_pred = dense_cnn(input, nclass)
        recognition_model = Model(input=input, outputs=y_pred)
        model_path = './recognition/...'
        recognition_model.load_weights(model_path)
        if os.path.exists(model_path):
            print('loading models')
        else:
            print('model do not exist')
            break

        j = 0
        txt_path = os.path.join(FLAGS.output_dir, im_fn.split('/')[-1].split('.')[0])
        with open('{}.txt'.format(txt_path), 'a', encoding='utf-8')as outf:
            for points in points_list:
                j += 1
                pt1 = points[0]
                pt2 = points[1]
                pt3 = points[2]
                pt4 = points[3]
                degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
                text_img = dumpRotateImage(im[:, :, ::-1], degree, pt1, pt2, pt3, pt4)
                text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)

                text_h, text_w = text_img.shape[:2]
                if text_h // text_w > 1:
                    continue
                dst_h = 32
                dst_w = text_w * dst_h // text_h
                text_img = cv2.resize(text_img, (dst_w, dst_h))
                X = text_img.reshape([1, 32, -1, 1])
                y_pred = recognition_model.predict(X)
                y_pred = y_pred[:, :, :]
                out = _decode(y_pred)
                img_PIL = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype('./utils/simsun.ttc', 12)
                fillColor = (255, 0, 0)
                draw = ImageDraw.Draw(img_PIL)
                if out is None:
                    out = ''
                draw.text(pt4, out, font=font, fill=fillColor)
                draw_img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                outf.write('{}. \t{}\n'.format(j, out))

            if not FLAGS.no_write_images:
                img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                cv2.imwrite(img_path, draw_img)

    print('total time = ', time.time()-t0)


if __name__ == '__main__':
    tf.app.run()
