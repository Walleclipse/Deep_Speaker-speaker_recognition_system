#  Deep Speaker: speaker recognition system

Data Set: [LibriSpeech](http://www.openslr.org/12/)  
Reference paper: "Deep Speaker: an End-to-End Neural Speaker Embedding System" https://arxiv.org/pdf/1705.02304.pdf  
Reference code : https://github.com/philipperemy/deep-speaker (Thanks Philippe Rémy. I have greatly modified the code during the experiment, but the theme is still similar.)  
  
This code was trained using librispeech-train-clean dataset, tested using librispeech-test-clean dataset. In my code librispeech dataset shows ~5% EER using CNN.   
  
## About Code
train.py  
This is the main file. This file train the model,then save the model and evaluate the result every specific steps.  
models.py  
This is the implementation of model used in this project. It contains three models, the CNN model (similar with the paper's CNN), the GRU model (similar with the paper's GRU), and the third model is simplified simple_cnn model.  
select_batch.py  
Choose the optimal batch feed to the network. This is one of the core of this experiment.   
triplet_loss.py  
This is the code for calculating the triplet-loss for network training.  
test_model.py  
This is a code that evaluate (test) the model, Such as eer...   
eval_matrics.py  
This file contains equal error rate, f-measure, accuracy and other metrics used in evaluation part. 
pretaining.py  
This is a code for pre-training of softmax classification.  
pre_process.py  
This code implemented for read the voice-data, filter the mute, extract the fbank feature, and save the extracted-features as .npy format.  
  
## Results  
This code was trained using librispeech-train-clean dataset, tested using librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER using CNN. 
  
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/loss.png"  width="400" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/EER.png" width="400" ></div>  
    
  If you want to know more details, please read 'deep_speaker实验报告.pdf'(Chinese). If you want to read details in English, please contact me.  
