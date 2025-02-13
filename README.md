# MUStroke

We use [MARLIN](https://github.com/ControlNet/MARLIN) for visual feature extraction, and pretrained VGG model from [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/tree/main) for audio feature extraction.

To install environment:

```
conda create -n DSV2 python=3.9
conda activate DSV2
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install torchaudio==2.2.0
pip install marlin_pytorch
```

To run code for uncertainty estimation experiment: 
```
python ablate_PEACE_MAE_uncertain.py --epoch 100
```

