from PIL import Image
import numpy as np


def get_patches(image_path, N):
    # read image
    img = Image.open(image_path)
    size = min(img.size)
    img = img.resize((size, size), Image.LANCZOS)
    # transform into numpy array
    img_array = np.array(img)
    # get size
    height, width, channels = img_array.shape
    # get the number of patches
    total_patches = N * N
    patches = []
    # get the size of each patch
    patch_height = height // N
    patch_width = width // N

    for i in range(N):
        for j in range(N):
            # get the starting position
            start_row = i * patch_height
            start_col = j * patch_width
            # make sure the image is inside the image boundary
            if start_row + patch_height <= height and start_col + patch_width <= width:
                patch = img_array[start_row:start_row + patch_height, start_col:start_col + patch_width]
                patches.append(patch)
    print(f"extract {len(patches)} patches。")

    for idx, patch in enumerate(patches):
        patch_image = Image.fromarray(patch)
        # transform into RGB type
        if patch_image.mode == 'RGBA':
            patch_image = patch_image.convert('RGB')

    return patches, patch_height

def convert_to_2d_list(flat_list, N):
    # make sure the length of array to N*N
    if len(flat_list) != N * N:
        raise ValueError(f"the length must be {N * N}")
    # create a N*N 2D array
    two_d_list = []
    for i in range(N):
        row = flat_list[i * N:(i + 1) * N]
        two_d_list.append(row)

    return two_d_list

def reconstruct_image(patches, patch_height, patch_width, N):
    # make sure the length of array to N*N
    if len(patches) != N*N:
        raise ValueError(f"the length must be {N*N}")
    a = patches[0]
    reconstructed_image = np.zeros((patch_height * N, patch_width * N, patches[0].shape[2]), dtype=patches[0].dtype)

    # put patches into the 2D array
    for i in range(N):
        for j in range(N):
            patch = patches[i * N + j]  # calculate the index
            reconstructed_image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = patch

    return reconstructed_image


def main(image_path, N, M, k1, k2):
    patches, patch_height = get_patches(image_path, N)
    patches_2d = convert_to_2d_list(patches, N)

    patch_sampleed = []
    for i in range(int(N/M)):
        for j in range(int(N/M)):
            patch_sampleed.append(patches_2d[k1+i*M][k2+j*M])

    reconstructed_image1 = reconstruct_image(patch_sampleed, patch_height, patch_height, int(N/M))
    print(type(reconstructed_image1))
    #  transform NumPy array into image
    reconstructed_image1 = Image.fromarray(reconstructed_image1)
    # save image
    # image.save('output_image.png')
    # transform RGB type
    if reconstructed_image1.mode == 'RGBA':
        reconstructed_image1 = reconstructed_image1.convert('RGB')

    reconstructed_image1.save()  # here should be your address

# your image address
image_path = r"D:\Learning_Rescoure\Project\3.Atrous_Mamba\Code\文章\sample_figure\figure3\original.png"
N = 128  # expected number of patches
M = 2  # interval

for n in [128, 64, 32, 16]:
    main(image_path, n, M, 0, 0)
    main(image_path, n, M, 0, 1)
    main(image_path, n, M, 1, 0)
    main(image_path, n, M, 1, 1)








