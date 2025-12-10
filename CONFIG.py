import argparse
seed = 700
import numpy as np
import random as rn
import os
import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
torch.cuda.empty_cache()
# rn.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
# 定义命令行参数
parser = argparse.ArgumentParser(description="输入参数")
parser.add_argument("--data_name",type=str,default="mscoco_deep")
parser.add_argument("--gpuIdx",type=int,default=0)
parser.add_argument("--batch_size",type=int,default=128)

# 新增参数
parser.add_argument("--hidden_dim",type=int,default=128)
parser.add_argument("--output_dim",type=int,default=128)
parser.add_argument("--alpha",type=float,default=0.9,help="weight of origial dech loss")
# beta 已经被定义（目前已经被删除）
parser.add_argument("--gamma",type=float,default=0.1,help="weight of fuzzy loss")
parser.add_argument("--eta",type=float,default=0.5, help="weight of consistency learning loss")
parser.add_argument('--delta', type=float, default=0.5)

parser.add_argument('--use_relu', type=bool, default=True, 
                    help='Whether to use ReLU to calculate membership degree in FuzzyLogicModule.')
parser.add_argument('--use_cl', type=bool, default=True, 
                    help='Whether to use the consistency learning (contrastive) loss.')

parser.add_argument("--bit",type=int,default=128)
parser.add_argument("--tau",type=float,default=0.2)
parser.add_argument("--max_epochs",type=int,default=200)
parser.add_argument("--is_eval",type=int,default=0)
parser.add_argument("--unc_val",type=float,default=0.3)
parser.add_argument("--anneal_coff",type=int,default=300)
parser.add_argument("--continue_eval",type=int,default=0)
parser.add_argument("--para",type=float,default=0.1)
parser.add_argument("--L3idx",type=int,default=5)
parser.add_argument("--hiden_layer",type=int,default=2)

args = parser.parse_args()


torch.cuda.set_device(args.gpuIdx)


model_save_dirname = f'./DECH__NEW/ckpt/'
os.makedirs(model_save_dirname,exist_ok=True)