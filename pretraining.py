
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance. Extract audio features and save it as npy file, cost 8443.623185157776 seconds
# test-clean: 40 speaker, 2620 utterance
# batchisize 32*3 : train on triplet: 3.3s/steps , softmax pre train: 3.1 s/steps

from models import convolutional_model, convolutional_model_simple
from glob import glob
import logging
import os
from keras.models import Model
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np
import random
import constants as c
import utils
from pre_process import data_catalog, preprocess_and_save
from select_batch import clipped_audio
from time import time
import sys
from sklearn.model_selection import train_test_split

def loadFromList(x_paths, batch_start, limit, labels_to_id, no_of_speakers, ):
    x = []
    y_ = []
    for i in range(batch_start, limit):
        x_ = np.load(x_paths[i])
        x.append(clipped_audio(x_))

        last = x_paths[i].split("/")[-1]
        y_.append(labels_to_id[last.split("-")[0]])

    x = np.asarray(x)

    y = np.eye(no_of_speakers)[y_]    #one-hot
    y = np.asarray(y)
    return x, y

def batchTrainingImageLoader(train_data, labels_to_id, no_of_speakers, batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH):
    paths = train_data
    L = len(paths)
    while True:
        np.random.shuffle(paths)
        batch_start = 0
        batch_end = batch_size

        while batch_end < L:
            x_train_t, y_train_t = loadFromList(paths, batch_start, batch_end, labels_to_id, no_of_speakers)
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(x_train_t)
            random.seed(randnum)
            random.shuffle(y_train_t)
            yield (x_train_t, y_train_t)
            batch_start += batch_size
            batch_end += batch_size

def batchTestImageLoader(test_data, labels_to_id, no_of_speakers, batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH):
    paths = test_data
    L = len(paths)
    while True:
        np.random.shuffle(paths)
        batch_start = 0
        batch_end = batch_size

        while batch_end < L:
            x_test_t, y_test_t = loadFromList(paths, batch_start, batch_end, labels_to_id, no_of_speakers)
            yield (x_test_t, y_test_t)
            batch_start += batch_size
            batch_end += batch_size

def split_data(files, labels, batch_size):
    test_size = max(batch_size/len(labels),0.05)
    train_paths, test_paths, y_train, y_test = train_test_split(files, labels, test_size=test_size, random_state=42)
    return train_paths, test_paths


def main():
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    # train_path = "/Users/walle/PycharmProjects/Speech/coding/deep-speaker-master/audio/LibriSpeechSamples/train-clean-100"
    train_path = c.DATASET_DIR

    libri = data_catalog(train_path)
    files = list(libri['filename'])
    labels1 = list(libri['speaker_id'])

    labels_to_id = {}
    id_to_labels = {}
    i = 0

    for label in np.unique(labels1):
        labels_to_id[label] = i
        id_to_labels[i] = label
        i += 1

    no_of_speakers = len(np.unique(labels1))

    train_data, test_data = split_data(files, labels1, batch_size)
    batchloader = batchTrainingImageLoader(train_data,labels_to_id,no_of_speakers, batch_size=batch_size)
    testloader = batchTestImageLoader(test_data, labels_to_id, no_of_speakers, batch_size=batch_size)
    test_steps = int(len(test_data)/batch_size)
    x_test, y_test = testloader.__next__()
    b = x_test[0]
    num_frames = b.shape[0]
    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info("x_shape:{0}, y_shape:{1}".format(x_test.shape, y_test.shape))

    base_model = convolutional_model(input_shape=x_test.shape[1:], batch_size=batch_size, num_frames=num_frames)
    x = base_model.output
    x = Dense(no_of_speakers, activation='softmax',name='softmax_layer')(x)

    model = Model(base_model.input, x)
    logging.info(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("printing format per batch:", model.metrics_names)
    # y_ = np.argmax(y_train, axis=0)
    # class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_), y_)

    grad_steps = 0
    last_checkpoint = utils.get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
    last_checkpoint = None
    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        logging.info('[DONE]')

    orig_time = time()

    while True:
        orig_time = time()
        x_train, y_train = batchloader.__next__()
        [loss, acc] = model.train_on_batch(x_train, y_train)  # return [loss, acc]
        logging.info('Train Steps:{0}, Time:{1:.2f}s, Loss={2}, Accuracy={3}'.format(grad_steps,time() - orig_time, loss,acc))

        with open(c.PRE_CHECKPOINT_FOLDER + "/train_loss_acc.txt", "a") as f:
            f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))

        if grad_steps % c.TEST_PER_EPOCHS == 0:
            losses = []; accs = []
            for ss in range(test_steps):
                [loss, acc] = model.test_on_batch(x_test, y_test)
                x_test, y_test = testloader.__next__()
                losses.append(loss); accs.append(acc)
            loss = np.mean(np.array(losses)); acc = np.mean(np.array(accs))
            print("loss", loss, "acc", acc)
            logging.info('Test the Data ---------- Steps:{0}, Loss={1}, Accuracy={2}, '.format(grad_steps,loss,acc))
            with open(c.PRE_CHECKPOINT_FOLDER + "/test_loss_acc.txt", "a") as f:
                f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))

        if grad_steps  % c.SAVE_PER_EPOCHS == 0:
            utils.create_dir_and_delete_content(c.PRE_CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.PRE_CHECKPOINT_FOLDER, grad_steps, loss))

        grad_steps += 1

if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()