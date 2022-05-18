# config.py

digitcfg = {
    'backbone': "resnet18",
    'num_classes': 11,
    'lr_steps': (10, 15),
    'epoch': 20,
    'images_per_epoch': 50000,
    'np_ratio': 3,
    'feature_maps': [7, 14, 28, 56], 
    'size': 224,
    'steps': [32, 16, 8, 4],
    'min_sizes': [99, 45, 21, 9],
    'max_sizes': [153, 99, 45, 21],
    'aspect_ratios': [[2], [2], [2], [2]],
    'num_boxes': [4, 4, 4, 4],
    'clip': False,
    'name': 'digit',
}
