import albumentations
import cv2

def get_augmentation(dataset_name):
    if dataset_name == 'digit':
        train_transforms = albumentations.Compose([
            albumentations.Affine(translate_percent=(-0.1, 0.1), rotate=(-5, 5)),
            albumentations.Normalize(mean=(0.485), std=(0.229), max_pixel_value=255.0, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='albumentations'))

        valid_transforms = albumentations.Compose([
            albumentations.Normalize(mean=(0.485), std=(0.229), max_pixel_value=255.0, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='albumentations'))

        return train_transforms, valid_transforms
    else:
        raise Exception('Invalid dataset')
    