from os import listdir
from os.path import join
import glob

import torch
import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, color_system='YCbCr'):
    img = Image.open(filepath).convert(color_system)
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, color_system='YCbCr',colab_path="./",pattern="*.jpg"):
        super(DatasetFromFolder, self).__init__()
        pattern_img = f"{image_dir}/{pattern}"
        self.image_filenames  = [x for x in glob.glob(f'{pattern_img}')]

        self.input_transform  = input_transform
        self.target_transform = target_transform
        self.colab_path       = colab_path

    def __getitem__(self, index):
        input_filename    = self.image_filenames[index]
        target_filename   = input_filename.replace(".2.lr.jpg", ".orginal.jpg").replace(".4.lr.jpg", ".orginal.jpg").replace(".8.lr.jpg", ".orginal.jpg").replace(".16.lr.jpg", ".orginal.jpg")

        input_image     = Image.open(input_filename).convert('YCbCr').split()
        input_image     = input_image.convert('YCbCr')
        input_image,_,_ = input_image.split()
        target          = Image.open(target_filename).convert('YCbCr').split()
        target          = target.convert('YCbCr')
        target,_,_      = target.split()

        # input_image = load_img(input_filename)
        # target      = load_img(target_filename)

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)

    def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(image)))

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                            .format(image.shape))

        r: torch.Tensor = image[..., 0, :, :]
        g: torch.Tensor = image[..., 1, :, :]
        b: torch.Tensor = image[..., 2, :, :]

        delta = .5
        y: torch.Tensor = .299 * r + .587 * g + .114 * b
        cb: torch.Tensor = (b - y) * .564 + delta
        cr: torch.Tensor = (r - y) * .713 + delta
        return torch.stack((y, cb, cr), -3)