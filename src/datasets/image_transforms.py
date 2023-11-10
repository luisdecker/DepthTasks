import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


class ImageTransformer(object):
    def __init__(self, split, augmentation=False):
        self.split = split
        self.augmentation = augmentation
        assert self.split in [
            "train",
            "test",
            "validation",
        ], f"{self.split} is a invalid split"

    def get_joint_transform(self):
        if self.split == "train":
            return transforms.Compose([Transform2Pil()])
        else:
            return transforms.Compose([Transform2Pil()])

    def get_img_transform(self):
        if self.split == "train":
            return transforms.Compose([ToTensor()])
        else:
            return transforms.Compose([ToTensor()])

    def get_augmentation_transform(self):
        if self.split == "train":
            return transforms.Compose([ImageAugmentation()])
        return transforms.Compose([])

    def get_transform(self):
        joint_transform = self.get_joint_transform()
        img_transform = self.get_img_transform()
        if self.augmentation:
            augmentation_transform = self.get_augmentation_transform()
            return transforms.Compose(
                [joint_transform, augmentation_transform, img_transform]
            )

        return transforms.Compose([joint_transform, img_transform])


class Transform2Pil(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage()

    def _is_pil_image(self, img):
        if accimage is not None:
            return isinstance(img, (Image.Image, accimage.Image))
        else:
            return isinstance(img, Image.Image)

    def __call__(self, data_item):
        for d in data_item:
            if d in ["image_l", "image_r"]:
                if not self._is_pil_image(data_item[d]):
                    data_item[d] = self.to_pil(data_item[d])

        return data_item


class ImageAugmentation(object):
    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = functional.adjust_brightness(pil, brightness)
        pil = functional.adjust_contrast(pil, contrast)
        pil = functional.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        for d in data_item:
            if d in ["image_l", "image_r"]:
                data_item[d] = self.adjust_pil(data_item[d])

        return data_item


class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        for d in data_item:
            if d in [
                "image_l",
                "image_r",
                "depth_l",
                "depth_r",
                "seg_l",
                "seg_r",
            ]:
                data_item[d] = self.totensor(data_item[d])

        return data_item
