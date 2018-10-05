# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s
# spk_ver_20180401_20180630_70_3_reseg_test: 7200 waves,952 speaker Extract audio features and save it as npy file, cost 236.61852288246155 seconds
# spk_ver_20180401_20180630_70_3_reseg_train: 70387 waves,09250 speaker. Extract audio features and save it as npy file, cost 1776.130244731903 seconds
# lls_85_70_libri410_fenbirec713_cmu_clsu_reseg_train: 302871 waves, 15137 speaker
import os
from glob import glob
from python_speech_features import fbank, delta
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool

import silence_detector
import constants as c
from constants import SAMPLE_RATE
from time import time

np.set_printoptions(threshold=np.nan)
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def VAD(audio):
    chunk_size = int(SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(20)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    audio = VAD(audio.flatten()[start_frame:])   #去掉前 0.2s 的 bit 提示音

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    return audio

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)   #filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)


def prep(spk2utt,utt2path,out_dir=c.DATASET_DIR,name='0'):
    start_time = time()
    i = 0
    for s2u in spk2utt:
        speaker = s2u.split()[0]
        utts = s2u.split()[1:]
        for utt in utts:
            i += 1
            orig_time = time()
            utt_id = utt.split('_')[:-1]  #utr2spk 中的 utt id 是'ZEBRA-KIDS0000000_1735129_26445a50743aa75d_00000 去掉后面的 _000
            utt_id = '_'.join(utt_id)
            filepath = utt2path[utt_id]
                                                      #为了统一成和librispeech 格式一致 speaker与utt 用 '-'分割 speaker内部就用'_'
            target_filepath = out_dir + speaker.replace('-','_') + '-' + utt_id.replace('-','_') + '.npy'
            if os.path.exists(target_filepath):
                if i % 10 == 0: print("task:{0} No.:{1} Exist File:{2}".format(name, i, filepath))
                continue
            raw_audio = read_audio(filepath)
            if np.count_nonzero(raw_audio) < 1.0 * SAMPLE_RATE:    #如果非静音部分小于 1s 则舍弃掉这个音频
                continue
            feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
            if feature.ndim != 3 or feature.shape[0] < c.NUM_FRAMES or feature.shape[1] != 64 or feature.shape[2] != 1:
                print('there is an error in file:',filepath)
                continue
            np.save(target_filepath, feature)
            if i % 100 == 0:
                print("task:{0} cost time per audio: {1:.3f}s No.:{2} File name:{3}".format(name, time() - orig_time, i, filepath))
    print("task %s runs %d seconds. %d files" %(name, time()-start_time,i))


def preprocess_and_save(kaldi_dir=c.WAV_DIR,out_dir=c.DATASET_DIR): #kaldi_dir='/Users/walle/PycharmProjects/Speech/coding/my_deep_speaker/audio/spk_ver_20180401_20180630_70_3_reseg_test'

    orig_time = time()
    with open(kaldi_dir+'/spk2utt','r') as f:
        spk2utt = f.readlines()

    with open(kaldi_dir+'/wav.scp','r') as f:
        wav2path = f.readlines()

    utt2path = {}
    for wav in wav2path:
        utt = wav.split()[0]
        path = wav.split()[1]
        utt2path[utt] = path

    no_spk = min(len(spk2utt), 5000)
    spk2utt = spk2utt[:no_spk]
    print("extract fbank from audio and save as npy, using multiprocessing pool........ ")
    num_proc = 5
    p = Pool(num_proc)
    patch = int(len(spk2utt)/num_proc)
    for i in range(num_proc):
        if i < num_proc - 1:
            _spk2utt = spk2utt[i*patch: (i+1)*patch]
        else:
            _spk2utt = spk2utt[i*patch:]
        print("task %s speakers num: %d" %(i, len(_spk2utt)))
        p.apply_async(prep, args=(_spk2utt,utt2path,out_dir,i))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

    print("Extract audio features and save it as npy file, cost {0} seconds".format(time()-orig_time))
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def test():
    filename = 'audio/LibriSpeechSamples/train-clean-100/19/227/19-227-0036.wav'
    raw_audio = read_audio(filename)
    print(filename)
    feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
    print(filename)

if __name__ == '__main__':
    #test()
    preprocess_and_save()