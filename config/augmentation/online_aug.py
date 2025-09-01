import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_transform_2d():
    data_transforms = {
        'train': A.Compose([
            A.Resize(128, 128, p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'val': A.Compose([
            A.Resize(128, 128, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'test': A.Compose([
            A.Resize(128, 128, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        )
    }
    return data_transforms

# def data_transform_2d():
#     data_transforms = {
#         'train': A.Compose([
#             A.Resize(128, 128, p=1),
#             A.Flip(p=0.75),
#             A.Transpose(p=0.5),
#             A.RandomRotate90(p=1),
#         ],
#             additional_targets={'image2': 'image', 'image3': 'image', 'mask2': 'mask', 'mask3': 'mask'}
#         ),
#         'val': A.Compose([
#             A.Resize(128, 128, p=1),
#         ],
#             additional_targets={'image2': 'image', 'image3': 'image', 'mask2': 'mask', 'mask3': 'mask'}
#         ),
#         'test': A.Compose([
#             A.Resize(128, 128, p=1),
#         ],
#             additional_targets={'image2': 'image', 'image3': 'image', 'mask2': 'mask', 'mask3': 'mask'}
#         )
#     }
#     return data_transforms

def data_normalize_2d(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
    )
    return data_normalize

def data_transform_aerial_lanenet(H, W):
    data_transforms = A.Compose([
            A.Resize(H, W, p=1),
            ToTensorV2()
        ])
    return data_transforms

