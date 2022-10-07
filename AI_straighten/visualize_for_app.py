from torch.utils.data import DataLoader
import numpy as np
import torch

import AI_straighten.model as model
import AI_straighten.config as config
from AI_straighten.grid_sample import grid_sample
from AI_straighten.dataset import Data

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # inp_pil = Image.fromarray(inp) 
    return inp


def predictStraighten(model, img_list):

    valid_dataset = Data(img_list)
    test_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)

    
    data = next(iter(test_loader))
    input_tensor = data[1].cpu()
    images = data[0].cpu()
    transformed_input_tensor = model.stn(input_tensor).cpu()
        
        
    transformered_images = grid_sample(images, transformed_input_tensor)

    numpy_transform_image = np.transpose(transformered_images.detach().numpy(), (0,2,3,1))
    return numpy_transform_image

if __name__ == '__main__':
    straighten_model = model.get_model(config.TRAINING_CONDITION)
    checkpoint = torch.load('./weight/outputs_transformer_bound_grid7/epoch_40.pth')
    straighten_model.load_state_dict(checkpoint)
    straighten_model.eval()






    

