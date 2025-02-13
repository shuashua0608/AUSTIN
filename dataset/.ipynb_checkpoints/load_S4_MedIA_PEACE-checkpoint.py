import os
from PIL import Image
import numpy as np
import random
import torch
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import KFold
import torch.utils.data
import torchvision.transforms as transforms
import copy

cate2label = {'nonstroke': 0, 'stroke': 1}


def LoadData(root, audio_root, spec_path, current_fold, totallist, 
             batchsize_train, batchsize_eval):
    train_dataset = VideoDataset(
        labelid=1,
        current=current_fold,
        video_root=root,
        audio_root=audio_root,
        spec_path = spec_path, 
        video_list=totallist,
        transform=transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                      ]))

    val_dataset = VideoDataset(
        labelid=0,
        current=current_fold,
        video_root=root,
        audio_root=audio_root,
        spec_path = spec_path, 
        video_list=totallist,
        transform=transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                      ]))

    test_dataset = VideoDataset(
        labelid=2,
        current=current_fold,
        video_root=root,
        audio_root=audio_root,
        spec_path = spec_path, 
        video_list=totallist,
        transform=transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                      ]))
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=5, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(1234))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=5, pin_memory=True, worker_init_fn=np.random.seed(1234))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=5, pin_memory=True, worker_init_fn=np.random.seed(1234))

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset



def LoadParameter(_structure, _parameterDir):
    checkpoint = torch.load(_parameterDir)
    model_state_dict = _structure.state_dict()
    for key in checkpoint:
        if ((key == 'fc.weight') | (key == 'fc.bias')):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = checkpoint[key]
    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()
    return model

# feature map w/ computing difference
def load_all(labelid, roundid, video_root, audio_root, spec_root, video_list):
    print(audio_root)
    imgs_first = list()
    wavs_first = list()
    specs_first = list()
    if labelid == 1:
        strokelist = video_list[roundid * 2]  # video_list: total_list in main
        nonstrokelist = video_list[roundid * 2 + 3]
    elif labelid == 0:
        strokelist = video_list[roundid * 2 + 1]
        nonstrokelist = video_list[roundid * 2 + 4]
    else:
        strokelist = video_list[roundid * 2 + 2]
        nonstrokelist = video_list[roundid * 2 + 5]
    # print(video_list)
    video_list = []
    if ".DS_Store" in strokelist:
        strokelist.remove(".DS_Store")
    for item in strokelist:
        video_list.append(item + " stroke")

    if ".DS_Store" in nonstrokelist:
        nonstrokelist.remove(".DS_Store")
    for item in nonstrokelist:
        video_list.append(item + " nonstroke")

    cnt = 0
    index = []
    video_names = []

    for line in video_list:
        video_label = line.split()
        video_name = video_label[0]  # name of video
        label = cate2label[video_label[1]]  # label of video
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        spec_path = os.path.join(spec_root, video_name)
        audio_path = os.path.join(audio_root, video_name)  # video_path is the path of each video
        img_lists = os.listdir(video_path)
        # print(img_lists)
        img_lists.sort(key=lambda x: int(x[:-4]))  # sort files by ascending
        # print(img_lists)
        img_count = len(img_lists)
        for clip in img_lists:
            imgs_first.append((os.path.join(video_path, clip), label))
            specs_first.append(os.path.join(spec_path, clip[:-4]+'.png'))
            wavs_first.append(os.path.join(audio_path, clip[:-4]+'.npy'))
            
        video_names.append(video_name)
        index.append(np.ones(img_count, dtype=np.int) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    return np.array(imgs_first,dtype=object), np.array(wavs_first,dtype=np.string_), np.array(specs_first,dtype=np.string_), index, video_list


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, labelid, current, video_root, audio_root, spec_path, video_list, transform=None):
        self.label = labelid
        self.roundid = current
        self.audio_root = audio_root
        self.spec_path = spec_path
        self.imgs_first, self.wavs_first, self.specs_first, self.index, self.name_list = load_all(labelid, current, video_root, audio_root, spec_path, video_list)
        # remain to optimize about the parameter length
        self.transform = transform

    # def _wav2fbank(self, filename):
    #     waveform = None
    #     while waveform is None:
    #         try:
    #             waveform, sr = torchaudio.load(filename)
    #         except:
    #             waveform = None
            
    #     waveform = waveform - waveform.mean()
        
    #     fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
    #                                               window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=100)
    #     del waveform
        
    #     target_length = 600
    #     n_frames = fbank.shape[0]
    #     p = target_length - n_frames

    #     # cut and pad
    #     if p > 0:
    #         m = torch.nn.ZeroPad2d((0, 0, 0, p))
    #         fbank = m(fbank)
    #     elif p < 0:
    #         fbank = fbank[0:target_length, :]
        
    #     return fbank
        
    
    def __getitem__(self, index):
        path_first, target_first = copy.deepcopy(self.imgs_first[index])
        img_first = None
        while img_first is None:
            try:
                img_first = np.load(path_first)
            except:
                img_first = None
        img_first = np.float16(img_first)
        filename = self.wavs_first[index]
        fbank = np.load(filename)
        # fbank = self._wav2fbank(filename).squeeze(0)
        # fbank = (fbank + 4.26) / (4.57 * 2)
        # print(fbank.shape)
        spec_first = Image.open(self.specs_first[index]).convert("RGB")
        spec_first = self.transform(spec_first)
        # path_first = None
        
        del path_first
        return img_first, spec_first, fbank, target_first, self.index[index]


    def __len__(self):
        return len(self.imgs_first)

    def get_name(self):
        return self.name_list


if __name__ == "__main__":
    pass