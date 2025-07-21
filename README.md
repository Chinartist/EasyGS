# Plug-and-Play 3DGS
## Install
```bash
pip install pycolmap==3.11
pip install wandb
```


## How to train with COLMAP format data
* You can fine tune the lr in the function `training_setup` in the file `gaussian_model.py`
``` python
from EasyGS import GSer,LearningRate,LossWeights
import numpy as np
import os
from tqdm import tqdm

import shutil
from argparse_dataclass import dataclass,ArgumentParser
@dataclass
class Options:
    data_dir: str = "/home/tangyuan/project/Rescale/data/R39_1752206941400/"
    instance_id: str = "Instance_5"
parser = ArgumentParser(Options)
if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.data_dir
    instance_id = args.instance_id
    colmap_path = f"{data_dir}"
    images_folder = f"{data_dir}/images"
    pretrained_path = f"{data_dir}/{instance_id}.ply"
    save_dir = f"{data_dir}/3dgs"
    alphas_folder = f"{data_dir}/masks"
    LearningRate = {k:0.0 for k in LearningRate}
    GS = GSer(colmap_path=colmap_path,
                pretrained_path=pretrained_path,images_folder=images_folder,alphas_folder=alphas_folder,
                save_dir=save_dir,iterations=2000,lr_args= LearningRate, loss_weights=LossWeights,init_degree=3,max_sh_degree=3,
                eval_interval=100,save_interval=100,
                eval_rate=1.0,
                enable_densification=False,
                enable_reset_opacity=False,
                wandb_project=None
                ) 
    GS.train()
    GS.pbar.stop()
       
```
