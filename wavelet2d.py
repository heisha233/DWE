import os
import numpy as np
from PIL import Image
import pywt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default="./GlaS/image")
    parser.add_argument('--L_path', default="./GlaS/L")
    parser.add_argument('--H_path', default="./GlaS/H")
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--if_RGB', default=True, type=bool)
    args = parser.parse_args()

    if not os.path.exists(args.L_path):
        os.makedirs(args.L_path, exist_ok=True)
    if not os.path.exists(args.H_path):
        os.makedirs(args.H_path, exist_ok=True)

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        L_path = os.path.join(args.L_path, i)
        H_path = os.path.join(args.H_path, i)

        image = Image.open(image_path)
        if not args.if_RGB:
            image = image.convert('L')
        image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 3:
            LL, LH, HL, HH = [], [], [], []
            for channel in range(3):
                _LL, (_LH, _HL, _HH) = pywt.dwt2(image[:, :, channel], args.wavelet_type)
                LL.append(_LL)
                LH.append(_LH)
                HL.append(_HL)
                HH.append(_HH)
            LL, LH, HL, HH = np.stack(LL, axis=2), np.stack(LH, axis=2), np.stack(HL, axis=2), np.stack(HH, axis=2)
        else:
            LL, (LH, HL, HH) = pywt.dwt2(image, args.wavelet_type)

        LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255
        LL = Image.fromarray(LL.astype(np.uint8))
        LL.save(L_path)

        LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
        HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
        HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

        merge1 = HH + HL + LH

        merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255
        merge1 = Image.fromarray(merge1.astype(np.uint8))
        merge1.save(H_path)