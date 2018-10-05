import logging
import os
import re
from glob import glob
import matplotlib.pyplot as plt
import constants as c



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/*.h5'.format(checkpoint_folder), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)

def plot_loss(file=c.CHECKPOINT_FOLDER+'/losses.txt'):
    step = []
    loss = []
    mov_loss = []
    ml = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           loss.append(float(line.split(",")[1]))
           if ml == 0:
               ml = float(line.split(",")[1])
           else:
               ml = 0.01*float(line.split(",")[1]) + 0.99*mov_loss[-1]
           mov_loss.append(ml)


    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.show()

def plot_loss_acc(file=c.PRE_CHECKPOINT_FOLDER+'/test_loss_acc.txt'):
    step = []
    loss = []
    acc = []
    mov_loss = []
    mov_acc = []
    ml = 0
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           loss.append(float(line.split(",")[1]))
           acc.append(float(line.split(",")[-1]))
           if ml == 0:
               ml = float(line.split(",")[1])
               mv = float(line.split(",")[-1])
           else:
               ml = 0.01*float(line.split(",")[1]) + 0.99*mov_loss[-1]
               mv = 0.01*float(line.split(",")[-1]) + 0.99*mov_acc[-1]
           mov_loss.append(ml)
           mov_acc.append(mv)

    plt.figure(1)
    plt.subplot(211)
    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Steps")
    plt.ylabel("Losses ")
    plt.subplot(212)
    p1, = plt.plot(step, acc)
    p2, = plt.plot(step, mov_acc)
    plt.legend(handles=[p1, p2], labels=['Accuracy', 'moving_average_accuracy'], loc='best')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy ")
    plt.show()

def plot_acc(file=c.CHECKPOINT_FOLDER+'/acc_eer.txt'):
    step = []
    eer = []
    fm = []
    acc = []
    mov_eer=[]
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           eer.append(float(line.split(",")[1]))
           fm.append(float(line.split(",")[2]))
           acc.append(float(line.split(",")[3]))
           if mv == 0:
               mv = float(line.split(",")[1])
           else:
               mv = 0.1*float(line.split(",")[1]) + 0.9*mov_eer[-1]
           mov_eer.append(mv)

    p1, = plt.plot(step, fm, color='black',label='F-measure')
    p2, = plt.plot(step, eer, color='blue', label='EER')
    p3, = plt.plot(step, acc, color='red', label='Accuracy')
    p4, = plt.plot(step, mov_eer, color='red', label='Moving_Average_EER')
    plt.xlabel("Steps")
    plt.ylabel("I dont know")
    plt.legend(handles=[p1,p2,p3,p4],labels=['F-measure','EER','Accuracy','moving_eer'],loc='best')
    plt.show()

def changefilename(path):
    files = os.listdir(path)
    for file in files:
        name=file.replace('-','_')
        lis = name.split('_')
        speaker = '_'.join(lis[:3])
        utt_id = '_'.join(lis[3:])
        newname = speaker + '-' +utt_id
        os.rename(path+'/'+file, path+'/'+newname)

def copy_wav(kaldi_dir,out_dir):
    import shutil
    from time import time
    orig_time = time()
    with open(kaldi_dir+'/utt2spk','r') as f:
        utt2spk = f.readlines()

    with open(kaldi_dir+'/wav.scp','r') as f:
        wav2path = f.readlines()

    utt2path = {}
    for wav in wav2path:
        utt = wav.split()[0]
        path = wav.split()[1]
        utt2path[utt] = path
    print(" begin to copy %d waves to %s" %(len(utt2path), out_dir))
    for i in range(len(utt2spk)):
        utt_id = utt2spk[i].split()[0].split('_')[:-1]  #utr2spk 中的 utt id 是'ZEBRA-KIDS0000000_1735129_26445a50743aa75d_00000 去掉后面的 _000
        utt_id = '_'.join(utt_id)
        speaker = utt2spk[i].split()[1]
        filepath = utt2path[utt_id]
                                                      #为了统一成和librispeech 格式一致 speaker与utt 用 '-'分割 speaker内部就用'_'
        target_filepath = out_dir + speaker.replace('-','_') + '-' + utt_id.replace('-','_') + '.wav'
        if os.path.exists(target_filepath):
            if i % 10 == 0: print(" No.:{0} Exist File:{1}".format(i, filepath))
            continue
        shutil.copyfile(filepath, target_filepath)

    print("cost time: {0:.3f}s ".format(time() - orig_time))