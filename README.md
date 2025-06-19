# Austin model for stroke detection

To install environment:

```
conda create -n DSV2 python=3.9
conda activate DSV2
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install torchaudio==2.2.0
pip install marlin_pytorch
```

## Main implementation:

```
python ds_peace_uncertain_save_sigma.py --epochs 100 --wi 2.0 --w True
```
wi: Test different weights in CE loss (nn.CrossEntropyLoss(weight=torch.tensor([self.wi, 1.0])))\
w: when False, the second term in uncertainty loss is set to be constant 1. 

## other benchmarks: 
We use [MARLIN](https://github.com/ControlNet/MARLIN) for visual feature extraction, and pretrained VGG model from [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/tree/main) for audio feature extraction.

To run code for uncertainty estimation experiment: 
```
python ablate_PEACE_MAE_uncertain.py --epoch 100
```

We use [Facexformer](https://github.com/Kartik-3004/facexformer) for visual feature extraction, and pretrained VGG model from [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/tree/main) for audio feature extraction.


To run code for experiment with facexformer and one-peace backbones: 
```
python fxf_AST.py --epoch 100
```
To run code for uncertainty estimation experiment with facexformer and one-peace backbones (without visualization yet): 
```
python fxf_AST_uncertain.py --epoch 100
```
