# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:45:00 2020

@author: sarna
"""

import os
import numpy as np
import librosa
import scipy
import torch

class ParticipantLab:
    """class for handling participant data"""

    m_timestep = 0
    a_timestep = 0
    m_features = 6
    n_mfcc = 30
    motion_ftrs = 48
    audio_split = 10
    
    def __rearrange(self, a, window, overlap):
        if(len(a.shape) == 2):
            l, f = a.shape
            shape = (int( (l-overlap)/(window-overlap) ), window, f)
            stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
        elif(len(a.shape) == 1):
            l = len(a)
            shape = (int( (l-overlap)/(window-overlap) ), window)
            stride = (a.itemsize*(window-overlap), a.itemsize)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)

    
    
    def __init__(self, name, folder, win_size, overlap, normalized=False):
        self.name = name
        self.a_folder = []
        self.m_folder = []
        if normalized:
            motion_folder = '../Norm_data'
        else:
            motion_folder = folder
        for i in range(1,3):
        	self.a_folder.append(folder + '/' + name + str(i) + '/SegAudio/')
        	self.m_folder.append(motion_folder + '/' + name + str(i) + '/SegMotion/')
        self.a_folder_s1 = folder + '/' + name + '1/SegAudio/'
        self.m_folder_s1 = motion_folder + '/' + name + '1/SegMotion/'
        self.a_folder_s2 = folder + '/' + name + '2/SegAudio/'
        self.m_folder_s2 = motion_folder + '/' + name + '2/SegMotion/'
        self.data_s1 = []
        self.data_s2 = []
        self.rawMdataX_s1 = []
        self.rawAdataX_s1 = []
        self.rawdataY_s1 = []
        self.rawMdataX_s2 = []
        self.rawAdataX_s2 = []
        self.rawdataY_s2 = []
        self.labels = []
        self.win_size = win_size # in seconds
        self.overlap = overlap
        

    def readRawAudioMotionData(self):
        # ************ UNDER TEST *********************
        sr_motion =  50.7 #55.0 #50.7
        old_sr = 50.7
        sr_audio = 22050
        win_motion = int(np.round(self.win_size*sr_motion))
        olp_motion = int(self.overlap*win_motion)
        self.m_timestep = win_motion
        
        # session 1
        if os.path.exists(self.a_folder_s1):
            a_files = os.listdir(self.a_folder_s1)
            
            fileNames = [a.split('.')[0] for a in a_files]
            
            m_files = [a + '.csv' for a in fileNames]
            #m_files = sorted(os.listdir(self.m_folder_s1))
            #import pdb; pdb.set_trace()
            self.rawMdataX_s1 = np.zeros((0, int(self.win_size*old_sr), 6), dtype=np.float32)
            self.rawAdataX_s1 = np.zeros((0, int(self.win_size*sr_audio)), dtype=np.float32)
            self.rawdataY_s1 = np.zeros((0, 1))
            for index in range(0,len(a_files)):
                # Audio
                audio, sr_audio = librosa.load(self.a_folder_s1 + a_files[index], sr=None)
                win_audio = int(self.win_size*sr_audio)
                hop_length_audio = int((1-self.overlap)*win_audio)
                a_matrix = librosa.util.frame(audio, win_audio, hop_length_audio)
                #print(np.shape(audio),a_matrix.shape)
                # Motion
                motion = np.loadtxt(self.m_folder_s1 + m_files[index], delimiter=',')

                f_matrix = self.__rearrange(motion, win_motion, olp_motion)
                f_matrix = f_matrix[:,:int(self.win_size*old_sr)+1,:]
                #print('Motion {}, Audio {}'.format(np.shape(f_matrix), a_matrix.shape))
                # Labelling and concat
                a=np.shape(a_matrix)[-1]
                m=len(f_matrix)
                row=0
                if(a>m):
                    row = m
                    a_matrix = a_matrix[:,:row]
                else:
                    row = a
                    f_matrix = f_matrix[:row,:,:]
                if self.name == 'Test':
                    label = ord(a_files[index][4])-97
                else:
                    label = ord(a_files[index][3])-97   
                Y = label*np.ones((row,1), dtype=np.uint8)
                self.rawMdataX_s1 = np.vstack((self.rawMdataX_s1, f_matrix))
                self.rawdataY_s1 = np.vstack((self.rawdataY_s1, Y))
                self.rawAdataX_s1 = np.vstack((self.rawAdataX_s1, a_matrix.T))
            self.rawdataY_s1 = np.float32(self.rawdataY_s1)
            self.rawMdataX_s1 = np.float32(self.rawMdataX_s1)
            self.rawAdataX_s1 = np.float32(self.rawAdataX_s1)
            
        # session2   
        if os.path.exists(self.a_folder_s2): 
            a_files = os.listdir(self.a_folder_s2)
            
            fileNames = [a.split('.')[0] for a in a_files]
            
            m_files = [a + '.csv' for a in fileNames]
             #sorted(os.listdir(self.m_folder_s2))
            self.rawMdataX_s2 = np.zeros((0, win_motion, 6), dtype=np.float32)
            self.rawAdataX_s2 = np.zeros((0, int(self.win_size*sr_audio)), dtype=np.float32)

            self.rawdataY_s2 = np.zeros((0, 1))
            for index in range(0,len(a_files)):
                # Audio
                audio, sr_audio = librosa.load(self.a_folder_s2 + a_files[index], sr=None)
                win_audio = int(self.win_size*sr_audio)
                hop_length_audio = int((1-self.overlap)*win_audio)

                a_matrix = librosa.util.frame(audio, win_audio, hop_length_audio)

                # Motion
                motion = np.loadtxt(self.m_folder_s2 + m_files[index], delimiter=',')
                f_matrix = self.__rearrange(motion, win_motion, olp_motion)
                # Labelling and concat
                a=np.shape(a_matrix)[-1]
                m=len(f_matrix)
                row=0
                if(a>m):
                    row = m
                    a_matrix = a_matrix[:,:row]
                else:
                    row = a
                    f_matrix = f_matrix[:row,:,:]
                if self.name == 'Test':
                    label = ord(a_files[index][4])-97
                else:
                    label = ord(a_files[index][3])-97 
                self.labels.append(label)
                Y = label*np.ones((row,1), dtype=np.uint8)
                self.rawMdataX_s2 = np.vstack((self.rawMdataX_s2, f_matrix))
                self.rawdataY_s2 = np.vstack((self.rawdataY_s2, Y))
                self.rawAdataX_s2 = np.vstack((self.rawAdataX_s2, a_matrix.T))

            self.rawdataY_s2 = np.float32(self.rawdataY_s2)
            self.rawMdataX_s2 = np.float32(self.rawMdataX_s2)
            self.rawAdataX_s2 = np.float32(self.rawAdataX_s2)
        
    def size_of_data(self):
        print(np.shape(self.data_s1))
        print(np.shape(self.data_s2))
        
    def get_labels(self):
        print(self.labels)
         