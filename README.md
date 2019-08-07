#  Deep Speaker: speaker recognition system

Data Set: [LibriSpeech](http://www.openslr.org/12/)  
Reference paper: [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf)  
Reference code : https://github.com/philipperemy/deep-speaker (Thanks to Philippe Rémy)  
  
This code was trained on librispeech-train-clean dataset, tested on librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER with CNN model.   
  
## About the Code
`train.py`
This is the main file, contains training, evaluation and save-model function  
`models.py`  
The neural network used for the experiment. This file contains three models, CNN model (same with the paper’s CNN), GRU model (same with the paper's GRU), simple_cnn model. simple_cnn model has similar performance with the original CNN model, but the number of trained parameter dropped from 24M to 7M. 
`select_batch.py`  
Choose the optimal batch feed to the network. This is one of the cores  of this experiment.   
`triplet_loss.py`  
This is a code to calculate triplet-loss for network training. Implementation is the same as paper.  
`test_model.py`  
This is a code that evaluates (test) the model, in terms of EER...   
`eval_matrics.py`  
For calculating equal error rate, f-measure, accuracy, and other metrics 
`pretaining.py`  
This is for pre-training on softmax classification loss.  
`pre_process.py`  
Load the utterance, filter out the mute, extract the fbank feature and save the module in .npy format. 
  
## Experimental Results  
This code was trained on librispeech-train-clean dataset, tested on librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER with CNN model. 
  
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/loss.png"  width="400" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system/raw/master/demo/EER.png" width="400" ></div>  

## More Details  
  If you want to know more details, please read [deep_speaker_report.pdf](deep_speaker_report.pdf) (English) or [deep_speaker实验报告.pdf](deep_speaker实验报告.pdf) (中文). 
