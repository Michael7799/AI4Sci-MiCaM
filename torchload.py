import torch

import os
import os.path as path
from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from model.mol_graph import MolGraph
from model.mydataclass import batch_train_data, mol_train_data, train_data
from model.vocab import SubMotifVocab

def view_pth_file(file_path):
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        print("Contents of the .pth file:")
        print(data)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Error while loading the .pth file: {e}")

if __name__ == "__main__":
    file_path = input("Enter the path to the .pth file: ")
    view_pth_file(file_path)
