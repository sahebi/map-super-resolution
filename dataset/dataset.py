from os import listdir
from os.path import join
import glob

import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, color_system='YCbCr'):
    img = Image.open(filepath).convert(color_system)
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, color_system='YCbCr',colab_path="./"):
        super(DatasetFromFolder, self).__init__()
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames  = [x for x in glob.glob(f'{image_dir}/*.lr.jpg')]

        self.input_transform  = input_transform
        self.target_transform = target_transform
        self.colab_path       = colab_path

    def __getitem__(self, index):
        # input_image = load_img(self.image_filenames[index])
        # target = input_image.copy()
        input_filename    = self.image_filenames[index].replace('./','/content/gdrive/MyDrive/super-resolution/map-super-resolution/')
        # target_filename   = input_filename.replace(".lr.jpg", ".orginal.jpg")
        target_filename   = input_filename.replace(".2.lr.jpg", ".x.orginal.jpg").replace(".4.lr.jpg", ".x.orginal.jpg").replace(".8.lr.jpg", ".x.orginal.jpg")


        input_image = load_img(input_filename)
        target      = load_img(target_filename)

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)
