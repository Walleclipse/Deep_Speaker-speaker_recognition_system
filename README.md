# Deep-Speaker-speaker-recognition-system

Data Set: librispeech  
Reference paper: "Deep Speaker: an End-to-End Neural Speaker Embedding System" https://arxiv.org/pdf/1705.02304.pdf  
Reference code : https://github.com/philipperemy/deep-speaker (Thanks philipperemy. I have greatly modified the code during the experiment, but the theme is still similar.)  
  
This code was trained using librispeech-train-clean dataset, tested using librispeech-test-clean dataset. In my code librispeech dataset shows ~5% EER using CNN.   
  
## About Code
Train.py  
This is the main file, which can be trained after running, and saves the model and test every time a certain number of steps.  
Models.py  
This is the module for creating the model. It consists of three models, the CNN model (consistent with the paper), the GRU model (consistent with the paper), and the third model is my own simplified simple_cnn model.  
Select_batch.py  
Choose the optimal batch feed to the network. This is one of the core of this experiment.   
Triplet_loss.py  
This is the module for calculating the triplet-loss for network training.  
Test_model.py  
This is a module that tests the model and tests parameters such as eer.   
Eval_matrics.py  
Input prediction and labels can be calculated, equal error rate, f-measure, accuracy and other indicators  
pertaining.py  
This is a module for pre-training of softmax classification.  
Pre_process.py  
This is to read the voice, filter the mute, extract the fbank feature, and save the module in .npy format.  
  
## Results  
This code was trained using librispeech-train-clean dataset, tested using librispeech-test-clean dataset. In my code, librispeech dataset shows ~5% EER using CNN.   
  
If you want to know more details, please read 'deep_speaker实验报告.pdf'(Chinese). If you want to read details in English ，please contact me.  
