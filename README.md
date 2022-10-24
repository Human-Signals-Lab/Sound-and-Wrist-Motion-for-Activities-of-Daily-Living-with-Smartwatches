# Sound-and-Wrist-Motion-for-Activities-of-Daily-Living-with-Smartwatches

This is the research repository for Leveraging Sound and Wrist Motion to Detect Activities of Daily Living with Commodity Smartwatches. It contains the deep learning pipeline to reproduce the results in the paper.


## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Dataset 

The dataset is made available for download on [Texas Data Repository Dataverse](https://doi.org/10.18738/T8/NNDFQD). Use of this dataset in publications must be acknowledged by referencing our [paper](#reference).


## Scripts 

### List of scripts:

- [ParticipantLab.py](ParticipantLab.py): includes the needed classes for loading the data while keeping track of participants, sessions, and activities. 
- [utils_.py](utils_.py): includes helper plotting functions.
- [utils](utils): includes several utils scripts for training, evaluation, and other loss related functions.
- [models.py](models.py): includes the neural networks, implemented using `pytorch`. The script includes all model configurations we experimented with.
- Leave-One-Participant-Out (LOPO) Evalution:
  - [main_LOPO_motion.py](main_LOPO_motion.py), [main_LOPO_Audio.py](main_LOPO_Audio.py), [main_LOPO_MotionAudio.py](main_LOPO_MotionAudio.py), [main_LOPO_MotionAudio_SoftmaxAveraging.py](main_LOPO_MotionAudio_SoftmaxAveraging.py): main scripts that run training as well as inference after training for Leave-One-Participant-Out (LOPO) evaluation for single-modal (motion/audio) and fusion-based models respectively.
- Leave-One-Session-Out (LOSO) Evalution:
  - [main_LOSO_motion.py](main_LOSO_motion.py), [main_LOSO_Audio.py](main_LOSO_Audio.py), [main_LOSO_MotionAudio.py](main_LOSO_MotionAudio.py), [main_LOSO_MotionAudio_SoftmaxAveraging.py](main_LOSO_MotionAudio_SoftmaxAveraging.py): main scripts that run training as well as inference after training for Leave-One-Session-Out (LOSO) evaluation for single-modal (motion/audio) and fusion-based models respectively.
- Personalized-LOPO (P-LOPO) Evalution:
  - [main_LOPO+1_motion.py](main_LOPO+1_motion.py), [main_LOPO+1_Audio.py](main_LOPO+1_Audio.py), [main_LOPO+1_MotionAudio.py](main_LOPO+1_MotionAudio.py), [main_LOPO+1_MotionAudio_SoftmaxAveraging.py](main_LOPO+1_MotionAudio_SoftmaxAveraging.py): main scripts that run training as well as inference after training for P-LOPO evaluation for single-modal (motion/audio) and fusion-based models respectively.
- **In-The-Wild** Analysis:
  - Inference: [main_WILD_Inference_Audio.py](main_WILD_Inference_Audio.py), [main_WILD_Inference_motion.py](main_WILD_Inference_motion.py), [main_WILD_Inference_AudioMotion.py](main_WILD_Inference_AudioMotion.py), [main_WILD_Inference_AudioMotion_PerParticipant.py](main_WILD_Inference_AudioMotion_PerParticipant.py), [main_Wild_Inference_AudioMotion_SoftmaxAveraging.py](main_Wild_Inference_AudioMotion_SoftmaxAveraging.py)
  - Finetune: [main_WILD_FinetuneInference_AudioMotion_PerParticipant.py](main_WILD_FinetuneInference_AudioMotion_PerParticipant.py) 

To run the scripts, check the next [section](#running-the-scripts).


## Model Definition:
Models defined in [models.py](models.py)
 
| Model Definition                                 | Input Modalities | Motion Model       | Audio Model | Fusion Method |
| ----------------                                 | :--------------: | :-----------:      | :---------: | :-----------: |
| AttendDiscriminate                               | Motion           | AttendDiscriminate | _           | _             |
| DeepConvLSTM_Classifier                          | Motion           | DeepConvLSTM       | _           | _             |
| Audio_CNN14                                      | Audio            | _                  | CNN14       | _             |
| AttendDiscriminate_MotionAudio_CNN14             | Motion + Audio   | AttendDiscriminate | CNN14       | Attention     |
| AttendDiscriminate_MotionAudio_CNN14_Concatenate | Motion + Audio   | AttendDiscriminate | CNN14       | Concatenate   |
| DeepConvLSTM_MotionAudio_CNN14_Attention         | Motion + Audio   | DeepConvLSTM       | CNN14       | Attention     |
| DeepConvLSTM_MotionAudio_CNN14_Concatenate       | Motion + Audio   | DeepConvLSTM       | CNN14       | Concatenate   |

### Running the Scripts:

As you can see above, python scripts are named according to which evaluation setting (LOPO, LOSO, or P-LOPO/LOPO+1) and which modalities as input (Audio, Motion for single-modal and MotionAudio for fusion). In each script, you define the corresponding model by changing the model definition in the script:
```
model_name = '{MODEL_DEFINITION}' # define one of the model definitions defined in the previous section
experiment = '{EXPERIMENT_NAME}'  # this will identify the name of the folder created where model + results are saved
``` 

Note that the model definition should be in line with the input modality of the script. Also, make sure the data is saved in a './Data' directory. 


That's all! For help, questions, and general feedback, contact Rebecca Adaimi (rebecca.adaimi@utexas.edu) or Sarnab Bhattacharya (sarnab2008@utexas.edu). 


## Reference 

BibTex Reference:

```
@article{10.1145/3534582,
author = {Bhattacharya, Sarnab and Adaimi, Rebecca and Thomaz, Edison},
title = {Leveraging Sound and Wrist Motion to Detect Activities of Daily Living with Commodity Smartwatches},
year = {2022},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {6},
number = {2},
url = {https://doi.org/10.1145/3534582},
doi = {10.1145/3534582},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {jul},
articleno = {42},
numpages = {28},
keywords = {Activities of Daily Living, Smartwatch, In-the-wild, Multimodal classification, Motion sensing, Audio Classification, Sound Sensing, Dataset, Wearable, Gesture Recognition, Human Activity Recognition}
}
```