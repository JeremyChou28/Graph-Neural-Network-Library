'''
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2023-02-09 21:49:25
'''
import torch

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
