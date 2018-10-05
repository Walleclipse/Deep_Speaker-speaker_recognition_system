#coding:utf8
__author__ = 'peach'
import math
import logging

class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1);
        self.logger = logging.getLogger('balloon_thrift')


    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil


    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power


if __name__ == '__main__':
    import io
    import os
    import librosa

    # ffmpeg -i biglittle.wav -f s16le -acodec pcm_s16le big.raw
    wav_fn = '/Users/walle/PycharmProjects/Speech/coding/my_deep_speaker/audio/spk_ver_20180401_20180630_70_3_reseg_test/wav' \
             '/spk_ver_20180401_20180630_70_3_reseg_testZEBRA_KIDS00000_110411652-ZEBRA_KIDS00000_110411652_ff3875f4fb3e5ef4.wav'
    wav = librosa.load(wav_fn, sr=None, mono=True)[0]
    for x in wav[:10]:
        print(x)

    # wav_data = bytearray(open(wav_fn, 'rb').read())
    # wav, sr = sf.read(io.BytesIO(wav_data), channels=1, samplerate=BalloonConfig.sample_freq, dtype='float32', subtype='PCM_16', format='RAW')


    sample_freq = 16000
    chunk_size = int(sample_freq*0.05) # 50ms
    index = 0
    sil_detector = SilenceDetector(15)
    sil_count = 0
    nonsil_count = 0
    while index + chunk_size < len(wav):
        if sil_detector.is_silence(wav[index: index+chunk_size]):
            sil_count += 1
            print ('is sil:', index*1.0/sample_freq)
        else:
            print('non-sil:', index*1.0/sample_freq)
            nonsil_count += 1
        index += chunk_size

    print('non sil count:',nonsil_count,' non sil length (s):',nonsil_count*0.05)
    print('sil count:',sil_count, 'sil length (s):',sil_count*0.05)
