from .nodes.PanoramaToCubemap import PanoramaToCubemap
from .nodes.CubemapToPanorama import CubemapToPanorama
NODE_CLASS_MAPPINGS = {
    "panorama2cubemap": PanoramaToCubemap,
    "cubemap2panorama": CubemapToPanorama 
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "Panorama2Cubemap": "Panorama 2 Cubemap",
    "Cubemap2Panorama": "Cubemap 2 Panorama"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
