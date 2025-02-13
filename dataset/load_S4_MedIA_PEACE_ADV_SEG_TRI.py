import sys
import os
import copy
from PIL import Image
import numpy as np
import random
import torch
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import KFold
import torch.utils.data
import torchvision.transforms as transforms
# torchaudio.set_audio_backend("soundfile")
cate2label = {'nonstroke': 0, 'stroke': 1}


def LoadData(root, audio_root, spec_path, current_fold, totallist, tri_dic,
             batchsize_train, batchsize_eval):
    train_dataset = VideoDataset(
        labelid=1,
        current=current_fold,
        video_root=root,
        audio_root=audio_root,
        spec_path = spec_path, 
        video_list=totallist,
        tri_dic = tri_dic,
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
        tri_dic = tri_dic,
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
        tri_dic = tri_dic,
        transform=transforms.Compose([
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                      ]))
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True, worker_init_fn=np.random.seed(1234))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=0, pin_memory=True, worker_init_fn=np.random.seed(1234))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=0, pin_memory=True, worker_init_fn=np.random.seed(1234))

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
def load_all(labelid, roundid, video_root, audio_root, spec_root, video_list,tri_dic):
    # print(audio_root)
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
        tri_label = int(tri_dic[video_name])
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
            imgs_first.append((os.path.join(video_path, clip), (label,tri_label)))
            specs_first.append(os.path.join(spec_path, clip[:-4]+'.png'))
            wavs_first.append(os.path.join(audio_path, clip[:-4]+'.npy'))
            
        video_names.append(video_name)
        index.append(np.ones(img_count, dtype=np.int) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    return np.array(imgs_first,dtype=object), np.array(wavs_first,dtype=np.string_), np.array(specs_first,dtype=np.string_), index, video_list


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, labelid, current, video_root, audio_root, spec_path, video_list, tri_dic, transform=None):
        self.label = labelid
        self.roundid = current
        self.audio_root = audio_root
        self.spec_path = spec_path
        self.imgs_first, self.wavs_first, self.specs_first, self.index, self.name_list = load_all(labelid, current, video_root, audio_root, spec_path, video_list,tri_dic)
        # print(self.imgs_first.shape,self.wavs_first.shape,self.specs_first.shape, self.index.shape)
        # remain to optimize about the parameter length
        self.transform = transform
        if self.label == 1:
            self.train_pairs_list = []
            print("Loading dataset...")
            with tqdm(total=len(self.imgs_first)) as pbar:
                for idx, (cur_frame, cur_wav, cur_spec) in enumerate(zip(self.imgs_first,self.wavs_first,self.specs_first)):
                    frame_index = self.index[idx]
                    choose_same = random.random() > 0.5
                    if choose_same:
                        frame_range = self.index == frame_index
                    else:
                        frame_range = self.index != frame_index
                    frame_list = self.imgs_first[frame_range]
                    wavs_list = self.wavs_first[frame_range]
                    spec_list = self.specs_first[frame_range]
                    chosen_idx = random.choice(range(len(frame_list)))
                    chosen_frame = frame_list[chosen_idx]
                    chosen_wav = wavs_list[chosen_idx]
                    chosen_spec = spec_list[chosen_idx]
                    # print(cur_frame, cur_wav, cur_spec,chosen_frame,chosen_wav,chosen_spec)
                    self.train_pairs_list.append((np.array([cur_frame, cur_wav, cur_spec,chosen_frame,chosen_wav,chosen_spec]), int(choose_same)))
                    pbar.update(1)
            print("Done")

    def __getitem__(self, index):
        if self.label == 1:
            path_first, is_same = copy.deepcopy(self.train_pairs_list[index])
            img_first = None
            while img_first is None:
                try:
                    img_first = np.load(path_first[0][0])
                except:
                    img_first = None
            img_first = np.float16(img_first)
            target_first = path_first[0][1][0]
            triage_first = path_first[0][1][1]
            
            spec_first = None
            while spec_first is None:
                try:
                    spec_first = Image.open(path_first[2]).convert("RGB")
                except:
                    print(path_first[2])
                    spec_first = None
                    
            spec_first = self.transform(spec_first)
            fbank = np.load(path_first[1])
            
            img_adv = None
            while img_adv is None:
                try:
                    img_adv = np.load(path_first[3][0])
                except:
                    print(path_first[3][0])
                    img_adv = None
            img_adv = np.float16(img_adv)
            spec_adv = None
            while spec_adv is None:
                try:
                    spec_adv = Image.open(path_first[5]).convert("RGB")
                except:
                    print(path_first[5])
                    spec_adv = None
            spec_adv = self.transform(spec_adv)
            
            fbank_adv = np.load(path_first[4])
            
            # path_first = None
            del path_first
            # print(img_first.dtype)
            # print(sys.getsizeof(img_first))
            return img_first, spec_first, fbank, target_first, triage_first, self.index[index], img_adv, spec_adv, fbank_adv, is_same
            
        else:
            path_first, (target_first,triage_first) = copy.deepcopy(self.imgs_first[index])
            img_first = None
            while img_first is None:
                try:
                    img_first = np.load(path_first)
                except:
                    print(path_first)
                    img_first = None
            img_first = np.float16(img_first)
            
            # TODO
            filename = self.wavs_first[index]
            fbank = np.load(filename)
            
            spec_first = None
            while spec_first is None:
                try:
                    spec_first = Image.open(self.specs_first[index]).convert("RGB")
                except:
                    print(self.specs_first[index])
                    spec_first = None
                    
            spec_first = self.transform(spec_first)
        
            del path_first
            return img_first, spec_first, fbank, target_first, triage_first, self.index[index]


    def __len__(self):
        return len(self.imgs_first)

    def get_name(self):
        return self.name_list
    

# def main():
#     arg_root = '../../Stroke_data/7seg/video_segment_fix/'  # '/Feature/Frames256/', /Feature/ori_large_frames/
#     # audio_root = '../Audio/new_trim/16k/cookie/'
#     audio_root = '../../Stroke_data/audio_old_2/segments/'
#     spec_path = '../../Stroke_data/audio_old_2/spec/'
#     strokes = []
#     nonstrokes = []
    
#     with open("../../Stroke/v3.csv","r") as f:
#         for vids in f.read().splitlines():
#             temp = vids.split(',')
#             if temp[1] == "N/A":
#                 continue
#             if temp[0] in ['0018','0147','0193','0119','0259','0274','0188','0227','0298']:
#                 continue
#             # if temp[0] not in orig_names:
#             #     continue
#             if temp[2] == '1':
#                 strokes.append(temp[0])
#             if temp[2] == '0':
#                 nonstrokes.append(temp[0])

#     print("num of stroke:", len(strokes))
#     print("num of nonstroke:", len(nonstrokes))

#     totallist = []
#     totallist.append([j for j in strokes if int(j) < 196])
#     totallist.append([j for j in strokes if (int(j) >= 196 and int(j) < 237)])
#     totallist.append([j for j in strokes if int(j) >= 237])
#     totallist.append([j for j in nonstrokes if int(j) < 196])
#     totallist.append([j for j in nonstrokes if (int(j) >= 196 and int(j) < 237)])
#     totallist.append([j for j in nonstrokes if int(j) >= 237])
    
#     arg_batchsize_train = 32
#     arg_batchsize_eval = 32
#     train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = LoadData(arg_root, audio_root,
#                                                                              spec_path, 0, totallist,
#                                                                              arg_batchsize_train,
#                                                                              arg_batchsize_eval)
#     count = 0
#     for epoch in range(3):
#         for i, (input_first, spec_img, fbank_cookie, target_first, index, input_adv, spec_adv, fbank_adv, target_adv) in enumerate(train_loader):
#             print(i)
#             pass

# if __name__ == "__main__":    
#     main()
    