import cv2  
from PIL import Image
from torchvision.transforms import Compose
from models import DPTSegmentationModel
from transforms import Resize, NormalizeImage, PrepareForNet
import numpy as np
import torch 
import torch.nn.functional as F

def write_segm_img(image, labels, alpha=0.5):
    """Returns segementation map overlayed on the image.

    Args:
        image (array): input image
        labels (array): labeling of the image
    """
    relevant_labels = [1 ,16,20,13]
    replaceable = {11:16, 24:20 ,31:20 ,34:16 , 46:16,54:1,60:1, 111:20,43:1 , 2:1 }
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i,  j ] not in relevant_labels:
                if replaceable.get(labels[i, j] ) == None:
                    labels[i ,j  ] = 0 
                else :
                    labels[i,j]  = replaceable[labels[i, j ]]
    
    mask = Image.fromarray(labels.squeeze().astype('uint8'))
    adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]
    mask.putpalette(adepallete)

    # img = Image.fromarray(np.uint8(255*image)).convert("RGBA")
    # seg = mask.convert("RGBA")
    # out = Image.blend(img, seg, alpha)

    return mask

def generatesegmap(img , model , device ,transform ):
    '''
    Generate the segmentation mask overlayed on image.
    Args:
        img: RGB image ndarray or Mat type 
        model: dpt model 
        device: cuda or cpu
        transform: Image processing operation beform sending it to model.
    
    Return RGBA PIL Image
    '''
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        out = model.forward(sample)

        prediction = torch.nn.functional.interpolate(
            out, size=img.shape[:2], mode="bicubic", align_corners=False
        )
        prediction = torch.argmax(prediction, dim=1) + 1
        prediction = prediction.squeeze().cpu().numpy()
    return write_segm_img(img, prediction, alpha=0.7)

if __name__ == "__main__":
    ''' Copy the code below in the main code before running the function.'''
    transform = Compose(
        [
            Resize(
                480,
                480,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preparing model 
    # Please download the model and place it in the directory. Link given below
    # https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-ade20k-53898607.pt
    model = DPTSegmentationModel(
            150,
            path="./dpt_hybrid-ade20k-53898607.pt", # Path where model is stored. Please change it accordingly.
            backbone="vitb_rn50_384",
        )
    model.eval()
    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
    model.to(device)
    
    ## After that call the function when needed 
    img = cv2.imread("./1.png")[:,:,::-1]/255 # replace the path or comment the line if not needed 
    # Please note that values in the image should be between 0 -1.
    
    out = generatesegmap(img , model , device ,transform )
    out.save("out.png")


