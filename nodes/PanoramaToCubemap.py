from PIL import Image
import numpy as np
import torch
  
from ..utils.ConvertCubemap import ConvertCubemap

class PanoramaToCubemap:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "panoramicImage": ("IMAGE", {}),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("Front+X","Right-Z","Back-X","Left+Z","Top+Y","Bottom-Y")
    FUNCTION = "panorama2cubemap"
    CATEGORY = "SkyBox_ToolBox/Convertors"
    
    def panorama2cubemap(self, panoramicImage):
        # Convert all input types to a numpy array for processing
        if isinstance(panoramicImage, Image.Image):
            # Convert PIL Image to numpy array
            pano_array = np.array(panoramicImage)
        elif isinstance(panoramicImage, torch.Tensor):
            # Convert tensor to numpy array
            pano_array = panoramicImage.numpy()
            if pano_array.ndim == 3 and pano_array.shape[0] == 3:  # CHW to HWC
                pano_array = pano_array.transpose(1, 2, 0)
        elif isinstance(panoramicImage, np.ndarray):
            pano_array = panoramicImage
        else:
            raise TypeError("Unsupported image type")

        # Process the panorama to cubemap
        face_size = panoramicImage.size()[1]  # Use image height for cube dimension to perseve the resulotion
        cubemap_faces = ConvertCubemap(pano_array, face_size).cubemap

        for x in range(len(cubemap_faces)):
            cubemap_faces[x] = torch.from_numpy(cubemap_faces[x])
            cubemap_faces[x] = cubemap_faces[x].unsqueeze(0)

        return tuple(cubemap_faces)

