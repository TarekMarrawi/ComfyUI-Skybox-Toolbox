import numpy as np
import cv2

class ConvertCubemap:
    cubemap = []
    def __init__(self, pano_array, face_size):
        """ Initialize the cubemap conversion with a panoramic image and face size. """
        self.cubemap = self.panorama_to_cubemap(pano_array, face_size)

    # Define the face transform constants
    face_transformations = np.array([
        [0, 0],             # Front face
        [np.pi / 2, 0],     # Right face
        [np.pi, 0],         # Back face
        [-np.pi / 2, 0],    # Left face
        [0, -np.pi / 2],    # Top face
        [0, np.pi / 2]      # Bottom face
    ])


    def create_cube_map_face(self, in_img, face_id=0, width=-1, height=-1):
        """ Create a single cube map face from a panoramic image. """
        in_height, in_width = in_img.shape[:2]

        if width == -1:
            width = in_width // 4
        if height == -1:
            height = width

        # Allocate map
        mapx = np.zeros((height, width), dtype=np.float32)
        mapy = np.zeros((height, width), dtype=np.float32)

        # Constants
        an = np.sin(np.pi / 4)
        ak = np.cos(np.pi / 4)

        ftu = self.face_transformations[face_id, 0]
        ftv = self.face_transformations[face_id, 1]

        # For each point in the target image,
        # calculate the corresponding source coordinates.
        for y in range(height):
            for x in range(width):

                # Map face pixel coordinates to [-1, 1] on plane
                nx = (y / height) - 0.5
                ny = (x / width) - 0.5

                nx *= 2
                ny *= 2

                # Map [-1, 1] plane coords to [-an, an]
                nx *= an
                ny *= an

                if ftv == 0:
                    # Center faces
                    u = np.arctan2(nx, ak)
                    v = np.arctan2(ny * np.cos(u), ak)
                    u += ftu
                elif ftv > 0:
                    # Bottom face
                    d = np.sqrt(nx * nx + ny * ny)
                    v = np.pi / 2 - np.arctan2(d, ak)
                    u = np.arctan2(ny, nx)
                else:
                    # Top face
                    d = np.sqrt(nx * nx + ny * ny)
                    v = -np.pi / 2 + np.arctan2(d, ak)
                    u = np.arctan2(-ny, nx)

                # Map from angular coordinates to [-1, 1], respectively.
                u = u / np.pi
                v = v / (np.pi / 2)

                # Warp around, if our coordinates are out of bounds.
                while v < -1:
                    v += 2
                    u += 1
                while v > 1:
                    v -= 2
                    u += 1

                while u < -1:
                    u += 2
                while u > 1:
                    u -= 2

                # Map from [-1, 1] to input texture space
                u = u / 2.0 + 0.5
                v = v / 2.0 + 0.5

                u = u * (in_width - 1)
                v = v * (in_height - 1)

                # Save the result for this pixel in map
                mapx[y, x] = u
                mapy[y, x] = v

        # Create the output face image
        face = cv2.remap(in_img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        return face

    def panorama_to_cubemap(self, pano_array, face_size):
        """ Convert a panoramic image to a cubemap format. """
        if isinstance(pano_array, np.ndarray):
            if pano_array.shape[0] == 1:
                pano_array = np.squeeze(pano_array, axis=0)
            if pano_array.ndim != 3 or pano_array.shape[2] != 3:
                raise ValueError("Expected a panoramic image with shape (height, width, 3)")
        else:
            raise ValueError("Input pano_array must be a 3D numpy array with shape (height, width, 3)")

        # Create the six cube faces
        faces = []
        for face_id in range(6):
            face = self.create_cube_map_face(pano_array, face_id=face_id, width=face_size, height=face_size)
            # Rotate the face based on the face ID to correct orientation
            if face_id == 0:  # Front
                pass
                face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                face = cv2.flip(face, 1)
            elif face_id == 1:  # Right
                face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                face = cv2.flip(face, 1)
            elif face_id == 2:  # Back
                face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                face = cv2.flip(face, 1)
            elif face_id == 3:  # Left
                face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                face = cv2.flip(face, 1)    
            elif face_id == 4:  # Top
                face = cv2.rotate(face, cv2.ROTATE_90_CLOCKWISE)
                face = cv2.flip(face, 1)
            elif face_id == 5:  # Bottom
                face = cv2.rotate(face, cv2.ROTATE_90_COUNTERCLOCKWISE)
                face = cv2.flip(face, 0)

            faces.append(face)

        return faces