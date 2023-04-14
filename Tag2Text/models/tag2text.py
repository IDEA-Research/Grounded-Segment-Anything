'''
 * Tag2Text
 * Written by Xinyu Huang
'''
import numpy as np
import json
import torch
import warnings

from torch import nn
from models.bert import BertConfig, BertModel, BertLMHeadModel
from models.vit import VisionTransformer
from models.swin_transformer import SwinTransformer

from models.utils import *

warnings.filterwarnings("ignore")

class Tag2Text_Caption(nn.Module):

    def __init__(self,
                 med_config='Tag2Text/configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list='Tag2Text/data/tag_list.txt'):
        r""" Tag2Text inference module, both captioning and tagging are included.
        Tag2Text is an efficient and controllable vision-language pre-training framework.
        Described in the paper "Tag2Text: Guiding Vision-Language Model via Image Tagging" https://arxiv.org/abs/2303.05657

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        super().__init__()

        # create image encoder
        if vit == 'swin_b':
            if image_size == 224:
                vision_config_path = 'Tag2Text/swin/config_swinB_224.json'
            elif image_size == 384:
                vision_config_path = 'Tag2Text/configs/swin/config_swinB_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # create tokenzier
        self.tokenizer = init_tokenizer()

        # Tag2Text employ encoder-decoder architecture for image-tag-text generation: image-tag interaction encoder and image-tag-text decoder
        # create image-tag interaction encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.tag_encoder = BertModel(config=encoder_config,
                                     add_pooling_layer=False)

        # create image-tag-text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file('Tag2Text/configs/q2l_config.json')
        q2l_config.encoder_width = vision_width
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Embedding(self.num_class, q2l_config.hidden_size)
        self.fc = GroupWiseLinear(self.num_class,
                                  q2l_config.hidden_size,
                                  bias=True)
        self.del_selfattention()

        # share weights of the lowest 2-layer of "image-tag interaction encoder" with the "image-tag recogntion decoder"
        tie_encoder_decoder_weights(self.tag_encoder, self.tagging_head, '',
                                    ' ')

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r') as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def generate(self,
                 image,
                 sample=False,
                 num_beams=3,
                 max_length=30,
                 min_length=10,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 tag_input=None,
                 return_tag_predict=False):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # if not user specified tags, recognized image tags using image-tag recogntiion decoder
        if tag_input == None:
            image_cls_embeds = image_embeds[:, 0, :]
            image_spatial_embeds = image_embeds[:, 1:, :]

            bs = image_spatial_embeds.shape[0]
            label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            tagging_embed = self.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            logits = self.fc(tagging_embed[0])

            targets = torch.where(
                torch.sigmoid(logits) > self.threshold,
                torch.tensor(1.0).to(image.device),
                torch.zeros(self.num_class).to(image.device))

            tag = targets.cpu().numpy()

            # delete some tags that may disturb captioning
            tag[:, self.delete_tag_index] = 0

            tag_input = []
            for b in range(bs):
                index = np.argwhere(tag[b] == 1)
                token = self.tag_list[index].squeeze(axis=1)
                tag_input.append(' | '.join(token))

        # beam search for text generation(default)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            tag_input_temp = []
            for tag in tag_input:
                for i in range(num_beams):
                    tag_input_temp.append(tag)
            tag_input = tag_input_temp

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # tokenizer input tags
        tag_input_tokenzier = self.tokenizer(tag_input,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=40,
                                             return_tensors="pt").to(
                                                 image.device)
        encoder_input_ids = tag_input_tokenzier.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # put input tag into image-tag interaction encoder to interact with image embeddings
        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # prompt trick for better captioning, followed BLIP
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search (default)
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        if return_tag_predict == True:
            if sample:
                return captions, tag_input
            else:
                return captions, tag_input[0:int(len(tag_input) / num_beams)]
        return captions


# load pretrained model parameters
def tag2text_caption(pretrained='', **kwargs):
    model = Tag2Text_Caption(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        print('vit:', kwargs['vit'])
        print('msg', msg)
    return model
