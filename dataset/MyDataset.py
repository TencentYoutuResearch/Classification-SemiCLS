""" The Code is under Tencent Youtu Public Rule
"""
from PIL import Image
from torch.utils.data import Dataset


#val，l-train dataset
class MyDataset(Dataset):
    """
    Interface provided for customized data sets

    names_file：a txt file, each line in the form of "image_path label"

    transform: transform pipline for mydataset

    """
    def __init__(self, names_file, transform=None):
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split(' ')[0]
        image = Image.open(image_path)
        if(image.mode == 'L'):
            image = image.convert('RGB')
        label = int(self.names_list[idx].split(' ')[1])

        if self.transform:
            image = self.transform(image)

        return image, label




