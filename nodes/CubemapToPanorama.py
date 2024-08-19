import torch
 
from ..utils.ConvertPanorama import ConvertPanorama

class CubemapToPanorama:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Front": ("IMAGE", {}),
                "Right": ("IMAGE", {}),
                "Back": ("IMAGE", {}),
                "Left": ("IMAGE", {}),
                "Top": ("IMAGE", {}),
                "Bottom": ("IMAGE", {}),
                "Width": ("INT", {"default": 3540, "min": 1}),
                "Height": ("INT", {"default": 1770, "min": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama_image",)
    FUNCTION = "cubemap2panorama"
    CATEGORY = "SkyBox_ToolBox/Convertors"

    def cubemap2panorama(self, Front, Right, Back, Left, Top, Bottom, Width, Height):
        #images = [Front, Right, Back, Left, Top, Bottom] // needs reorder as the next line does
        images = [Right, Bottom, Left, Back, Front, Top]

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU instead.")

        # Convert images to tensors
        images_tensors = [torch.tensor(img).to(device) for img in images]

        # Stack images into a single tensor
        images_tensor = torch.stack(images_tensors)
        images_tensor = images_tensor.squeeze(1)  # This removes the dimension at index 1

        # Reorder dimensions from (6, 512, 512, 3) to (6, 3, 512, 512)
        images_tensor = images_tensor.permute(0, 3, 1, 2)

        c2e = ConvertPanorama(512, 1024, 256, False)
        equi = c2e(images_tensor) 

        equi = equi[0, ...].permute(1, 2, 0).cpu().numpy()

        equi_image = torch.from_numpy(equi)
        equi_image = equi_image.unsqueeze(0)
        
        return (equi_image,)
