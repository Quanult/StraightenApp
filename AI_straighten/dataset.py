
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader



def make_mask(img):
    im0 = cv2.blur(img, (1,1))
    ret,thresh1 = cv2.threshold(im0, 2,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    bg= cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    mask = bg.clip(0,1)
    img_denoising = (img * mask)
    img_color = cv2.cvtColor(img_denoising, cv2.COLOR_GRAY2BGR)
    return img_color, bg

class Data(Dataset):
    def __init__(self, file_list):
        super(Data, self).__init__()
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        img, thresh1 = make_mask(img)
        img = np.transpose(img, (2,0,1))
        thresh1 = np.array([thresh1])
        
        return torch.tensor(img, dtype=torch.float), torch.tensor(thresh1, dtype=torch.float)


