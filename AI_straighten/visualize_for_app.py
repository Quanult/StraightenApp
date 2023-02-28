from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from PIL import Image

import AI_straighten.model as model
import AI_straighten.config as config
from AI_straighten.grid_sample import grid_sample
from AI_straighten.dataset import Data

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # inp_pil = Image.fromarray(inp) 
    return inp


def predictStraighten(model, img_list, phase = 'single_imge'):
    if phase == 'single_imge':
        # print(len(img_list))
        valid_dataset = Data(img_list)
        test_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)
        data = next(iter(test_loader))
        input_tensor = data[1].cpu()
        # print(input_tensor.shape)
        images = data[0].cpu()
        
        transformed_input_tensor = model.stn(input_tensor).cpu()   
        transformered_images = grid_sample(images, transformed_input_tensor)
        numpy_transform_image = np.transpose(transformered_images.detach().numpy(), (0,2,3,1))



#########
        # sample_img = cv2.imread('nice.jpg')
        # sample_img = np.transpose(sample_img, (2,0,1))
        # sample_data =  torch.tensor(np.expand_dims(sample_img, 0), dtype= torch.float).cpu()
        # # print(sample_data.shape)
        # # print(transformered_images.shape)
        # grid2 = model.stn(transformered_images[:,0,:,:]).cpu()
        # trans_img = grid_sample(sample_data, grid2)
        # np_sample_data = np.transpose(trans_img.detach().numpy(), (0,2,3,1))
        # cv2.imwrite('nice2.jpg', np_sample_data[0])

#########
        return numpy_transform_image


    else:
        data_black = torch.tensor(np.expand_dims(img_list[0], 1), dtype= torch.float).cpu()
        data_white = torch.tensor(np.expand_dims(img_list[1], 1), dtype= torch.float).cpu()
        output_grid = model.stn(data_black).cpu()

        out_black = grid_sample(data_black, output_grid)
        out_black = torch.squeeze(out_black, dim=1)
        out_black = out_black.detach().numpy()

        out_white = grid_sample(data_white, output_grid, padding_mode = 'border')
        out_white = torch.squeeze(out_white, dim=1)
        out_white = out_white.detach().numpy()

        for idx, image in enumerate(out_white):
            thresh1 = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)[1]
            thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            thresh2 = cv2.threshold(thresh1, 254, 255, cv2.THRESH_BINARY_INV)[1]
            thresh2 = thresh2.clip(0,1)
            image = image * thresh2
            image = image + thresh1
            out_white[idx] = image
        return out_black, out_white
     
if __name__ == '__main__':
    straighten_model = model.get_model(config.TRAINING_CONDITION)
    checkpoint = torch.load('./weight/outputs_transformer_bound_grid7/epoch_40.pth')
    straighten_model.load_state_dict(checkpoint)
    straighten_model.eval()






    

