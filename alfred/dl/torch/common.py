"""
Common utility of pytorch

this contains code that frequently used while writing torch applications

"""
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')