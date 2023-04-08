import cv2
import os
import numpy as np
from skimage.filters import gaussian


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    # changed = cv2.resize(changed, (512, 512))
    return changed

#
# def lip(image, parsing, part=17, color=[230, 50, 20]):
#     b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
#     tar_color = np.zeros_like(image)
#     tar_color[:, :, 0] = b
#     tar_color[:, :, 1] = g
#     tar_color[:, :, 2] = r
#
#     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     il, ia, ib = cv2.split(image_lab)
#
#     tar_lab = cv2.cvtColor(tar_color, cv2.COLOR_BGR2Lab)
#     tl, ta, tb = cv2.split(tar_lab)
#
#     image_lab[:, :, 0] = np.clip(il - np.mean(il) + tl, 0, 100)
#     image_lab[:, :, 1] = np.clip(ia - np.mean(ia) + ta, -127, 128)
#     image_lab[:, :, 2] = np.clip(ib - np.mean(ib) + tb, -127, 128)
#
#
#     changed = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
#
#     if part == 17:
#         changed = sharpen(changed)
#
#     changed[parsing != part] = image[parsing != part]
#     # changed = cv2.resize(changed, (512, 512))
#     return changed


if __name__ == '__main__':
    # 1  face
    # 10 nose
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair
    num = 116
    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }
    image_path = '/home/zll/data/CelebAMask-HQ/test-img/{}.jpg'.format(num)
    parsing_path = 'res/test_res/{}.png'.format(num)

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = np.array(cv2.imread(parsing_path, 0))
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]
    # colors = [[20, 20, 200], [100, 100, 230], [100, 100, 230]]
    colors = [[100, 200, 100]]
    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)
    cv2.imwrite('res/makeup/116_ori.png', cv2.resize(ori, (512, 512)))
    cv2.imwrite('res/makeup/116_2.png', cv2.resize(image, (512, 512)))

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    cv2.imshow('color', cv2.resize(image, (512, 512)))

    # cv2.imshow('image', ori)
    # cv2.imshow('color', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()















