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



To run the scripts with required arguments, check the next [section](#running-the-main-scripts).


### Running the Scripts:


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