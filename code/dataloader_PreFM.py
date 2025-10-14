import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import math


class PreFM_UnAV_dataset(Dataset):

    def __init__(self, mode, pd_dir, audio_dir, visual_dir, label_dir, f_label_dir, f_lens):
        self.mode = mode
        self.video_list = pd.read_csv(pd_dir, header=0, sep=',')
        self.video_name = self.video_list["filename"]
        self.video_num = len(self.video_name)
        self.curr_m = 10

        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        self.label_dir = label_dir
        self.f_label_dir = f_label_dir
        self.f_lens = f_lens

        self._init_dataset()


    def _init_dataset(self):
        self.inputs = []
        for i in range(self.video_num):
            sample = self.video_list.loc[i,:]
            sample_name = sample[0]
            sample_length = sample[1]

            if self.mode != 'train': 
                for i in range(60): # --val_batch_size, to facilitate the calculation of event level metrics
                    idx = i if i < sample_length else sample_length - 1
                    self.inputs.append([sample_name, sample_length, idx])

            else:
                curr_idx_num = math.ceil((sample_length - self.f_lens) / self.curr_m)
                for i in range(curr_idx_num):
                    start = i * self.curr_m
                    end = min((i + 1) * self.curr_m, sample_length-self.f_lens+1) 
                    random_point = np.random.randint(start, end - 1)
                    self.inputs.append([sample_name, sample_length, random_point])


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        name, length, curr_idx = self.inputs[idx]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy')) # 6.4*T 768
        visual = np.load(os.path.join(self.visual_dir, name + '.npy')) # T 768
        label = np.load(os.path.join(self.label_dir, name + '.npy')) # T 100
        f_label = np.load(os.path.join(self.f_label_dir, name + '.npy')) # T 1536
        
        # padding
        audio_pad = np.zeros((64,768),dtype=np.float32)
        visual_pad = np.zeros((10,768),dtype=np.float32)
        label_pad = np.zeros((10,100),dtype=np.float32)
        f_label_pad = np.zeros((10,1536),dtype=np.float32)
        audio = np.concatenate((audio_pad, audio), axis=0)
        visual = np.concatenate((visual_pad, visual), axis=0)
        label = np.concatenate((label_pad, label), axis=0)
        f_label = np.concatenate((f_label_pad, f_label), axis=0)


        # sample
        sample = {}
        sample['name'] = name
        sample['length'] = length
        audio_index = int((curr_idx+1)*6.4)
        sample['audio'] = audio[audio_index:audio_index+64,:]
        sample['visual'] = visual[curr_idx+1:curr_idx+11,:]
        sample['curr_label'] = label[curr_idx+1:curr_idx+11,:]
        sample['curr_f_label'] = f_label[curr_idx+1:curr_idx+11,:]

        if self.mode == 'train':
            
            sample['future_label'] = label[curr_idx+11:curr_idx+11+self.f_lens,:]
            sample['future_f_label'] = f_label[curr_idx+11:curr_idx+11+self.f_lens,:]

        return sample
    


class PreFM_plus_UnAV_dataset(Dataset):

    def __init__(self, mode, pd_dir, audio_dir, visual_dir, label_dir, f_label_dir, f_lens):
        self.mode = mode
        self.video_list = pd.read_csv(pd_dir, header=0, sep=',')
        self.video_name = self.video_list["filename"]
        self.video_num = len(self.video_name)
        self.curr_m = 30

        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        self.label_dir = label_dir
        self.f_label_dir = f_label_dir
        self.f_lens = f_lens

        self._init_dataset()


    def _init_dataset(self):

        self.inputs = []
        for i in range(self.video_num):
            sample = self.video_list.loc[i,:]
            sample_name = sample[0]
            sample_length = sample[2]

            if self.mode != 'train': 
                for i in range(240): # --val_batch_size, to facilitate the calculation of event level metrics
                    idx = i if i < sample_length else sample_length - 1
                    self.inputs.append([sample_name, sample_length, idx])

            else:
                curr_idx_num = math.ceil((sample_length - self.f_lens) / self.curr_m)
                for i in range(curr_idx_num):
                    start = i * self.curr_m
                    end = min((i + 1) * self.curr_m, sample_length-self.f_lens+1) 
                    random_point = np.random.randint(start, end - 1)
                    self.inputs.append([sample_name, sample_length, random_point])


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        name, length, curr_idx = self.inputs[idx]
        audio = np.load(os.path.join(self.audio_dir, name + '_one_peace_audio.npy')) # T 1536
        visual = np.load(os.path.join(self.visual_dir, name + '_one_peace_video_finetune.npy')) # T 1536
        visual = visual.astype(np.float32) # double-->float
        label = np.load(os.path.join(self.label_dir, name + '.npy')) # T 100
        f_label = np.load(os.path.join(self.f_label_dir, name + '.npy')) # T 1536
        
        # padding
        audio_pad = np.zeros((self.curr_m,1536),dtype=np.float32)
        visual_pad = np.zeros((self.curr_m,1536),dtype=np.float32)
        label_pad = np.zeros((self.curr_m,100),dtype=np.float32)
        f_label_pad = np.zeros((self.curr_m,1536),dtype=np.float32)

        audio = np.concatenate((audio_pad, audio), axis=0)
        visual = np.concatenate((visual_pad, visual), axis=0)
        label = np.concatenate((label_pad, label), axis=0)
        f_label = np.concatenate((f_label_pad, f_label), axis=0)


        # sample
        sample = {}
        sample['name'] = name
        sample['length'] = length
        sample['audio'] = audio[curr_idx+1:curr_idx+1+self.curr_m,:]
        sample['visual'] = visual[curr_idx+1:curr_idx+1+self.curr_m,:]
        sample['curr_label'] = label[curr_idx+1:curr_idx+1+self.curr_m,:]
        sample['curr_f_label'] = f_label[curr_idx+1:curr_idx+1+self.curr_m,:]

        if self.mode == 'train':
            
            sample['future_label'] = label[curr_idx+1+self.curr_m:curr_idx+1+self.curr_m+self.f_lens,:]
            sample['future_f_label'] = f_label[curr_idx+1+self.curr_m:curr_idx+1+self.curr_m+self.f_lens,:]

        return sample
    




class PreFM_LLP_dataset(Dataset):

    def __init__(self, mode, pd_dir, audio_dir, visual_dir, st_dir, label_dir, f_label_dir, f_lens):
        self.mode = mode
        self.video_list = pd.read_csv(pd_dir, header=0, sep=',')
        self.video_name = self.video_list["filename"]
        self.video_num = len(self.video_name)
        self.curr_m = 10

        self.audio_dir = audio_dir
        self.visual_dir = visual_dir
        self.st_dir = st_dir
        self.label_dir = label_dir
        self.f_label_dir = f_label_dir
        self.f_lens = f_lens

        self._init_dataset()


    def _init_dataset(self):
        self.inputs = []
        for i in range(self.video_num):
            video = self.video_list.loc[i,:]
            sample_name = video[2]
            sample_length = video[1] * 10

            if self.mode == 'train': 
                curr_idx_num = math.ceil((sample_length - self.f_lens) / self.curr_m)
                for i in range(curr_idx_num):
                    start = i * self.curr_m
                    end = min((i + 1) * self.curr_m, sample_length-self.f_lens+1) 
                    random_point = np.random.randint(start, end - 1)
                    event_label = video[i+3].split(",")
                    self.inputs.append([sample_name, sample_length, random_point, event_label])
                        
            else:      
                for i in range(60): # --val_batch_size, to facilitate the calculation of event level metrics
                    idx = i if i < sample_length else sample_length - 1
                    self.inputs.append([sample_name, sample_length, idx, ['none']])



    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample_name, sample_length, curr_idx, event_label = self.inputs[idx]

        audio = np.load(os.path.join(self.audio_dir, sample_name + '.npy')) # 6.4*T 768
        visual = np.load(os.path.join(self.visual_dir, sample_name + '.npy')) # T 768
        st = np.load(os.path.join(self.st_dir, sample_name + '.npy')) # T 512
        label_a = np.load(os.path.join(self.label_dir, 'audio', sample_name + '.npy')) # T 25
        label_v = np.load(os.path.join(self.label_dir, 'visual', sample_name + '.npy')) # T 25
        f_label_a = np.load(os.path.join(self.f_label_dir, 'audio', sample_name + '.npy')) # T 1536
        f_label_v = np.load(os.path.join(self.f_label_dir, 'visual', sample_name + '.npy')) # T 1536

        # padding
        audio_pad = np.zeros((64,768),dtype=np.float32)
        visual_pad = np.zeros((10,768),dtype=np.float32)
        st_pad = np.zeros((10,512),dtype=np.float32)
        label_a_pad = np.zeros((10,25),dtype=np.float32)
        label_v_pad = np.zeros((10,25),dtype=np.float32)
        f_label_a_pad = np.zeros((10,1536),dtype=np.float32)
        f_label_v_pad = np.zeros((10,1536),dtype=np.float32)

        audio = np.concatenate((audio_pad, audio), axis=0)
        visual = np.concatenate((visual_pad, visual), axis=0)
        st = np.concatenate((st_pad, st), axis=0)
        label_a = np.concatenate((label_a_pad, label_a), axis=0)
        label_v = np.concatenate((label_v_pad, label_v), axis=0)
        f_label_a = np.concatenate((f_label_a_pad, f_label_a), axis=0)
        f_label_v = np.concatenate((f_label_v_pad, f_label_v), axis=0)

        # sample
        sample={}

        sample['name'] = sample_name
        sample['length'] = sample_length
        audio_index = int((curr_idx+1)*6.4)
        sample['audio'] = audio[audio_index:audio_index+64,:]
        sample['visual'] = visual[curr_idx+1:curr_idx+11,:]
        sample['st'] = st[curr_idx+1:curr_idx+11,:]
        sample['curr_label_a'] = label_a[curr_idx+1:curr_idx+11,:]
        sample['curr_f_label_a'] = f_label_a[curr_idx+1:curr_idx+11,:]
        sample['curr_label_v'] = label_v[curr_idx+1:curr_idx+11,:]
        sample['curr_f_label_v'] = f_label_v[curr_idx+1:curr_idx+11,:]

        if self.mode == 'train':
            sample['futu_label_a'] = label_a[curr_idx+11:curr_idx+11+self.f_lens,:]
            sample['futu_f_label_a'] = f_label_a[curr_idx+11:curr_idx+11+self.f_lens,:]
            sample['futu_label_v'] = label_v[curr_idx+11:curr_idx+11+self.f_lens,:]
            sample['futu_f_label_v'] = f_label_v[curr_idx+11:curr_idx+11+self.f_lens,:]


        return sample