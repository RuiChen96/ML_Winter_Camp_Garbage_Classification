import os
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import make_dataset

from tnn.datasets.dataloader import sDataLoader

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def im_loader(path):
    return Image.open(path).convert('RGB')


class GarbageTestSet(torch.utils.data.Dataset):

    def __init__(self, root, inp_size):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)

        self.root = root
        self.imgs = imgs
        self.classes = classes

        self.transform = transforms.Compose([
            transforms.CenterCrop(inp_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        path, target = self.imgs[i]
        img = im_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, path, target


def get_loader(data_dir, inp_size, batch_size, shuffle=True, num_workers=3):

    dataset = GarbageTestSet(data_dir, inp_size)

    data_loader = sDataLoader(dataset, batch_size, shuffle, num_workers=num_workers)
    return data_loader
