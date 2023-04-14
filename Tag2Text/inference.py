'''
 * Tag2Text
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch
import torchvision.transforms as transforms

from PIL import Image
from models.tag2text import tag2text_caption

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/1641173_2291260800.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/tag2text_swin_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')
parser.add_argument('--thre',
                    default=0.68,
                    type=float,
                    metavar='N',
                    help='threshold value')
parser.add_argument('--specified-tags',
                    default='None',
                    help='User input specified tags')


def inference(image, model, input_tag="None"):

    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)

    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        return tag_predict[0], None, caption[0]

    # If user input specified tags:
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',', ' | '))

        with torch.no_grad():
            caption, input_tag = model.generate(image,
                                                tag_input=input_tag_list,
                                                max_length=50,
                                                return_tag_predict=True)

        return tag_predict[0], input_tag[0], caption[0]


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), normalize
    ])

    # delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
    delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]

    #######load model
    model = tag2text_caption(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_b',
                             delete_tag_index=delete_tag_index)
    model.threshold = args.thre  # threshold for tagging
    model.eval()

    model = model.to(device)
    raw_image = Image.open(args.image).resize(
        (args.image_size, args.image_size))
    image = transform(raw_image).unsqueeze(0).to(device)

    res = inference(image, model, args.specified_tags)
    print("Model Identified Tags: ", res[0])
    print("User Specified Tags: ", res[1])
    print("Image Caption: ", res[2])
