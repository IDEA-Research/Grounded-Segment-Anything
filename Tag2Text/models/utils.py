import os
import json
import torch
import math

from torch import nn
from typing import List
from transformers import BertTokenizer
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.vit import interpolate_pos_embed
from models.swin_transformer import interpolate_relative_pos_embed


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module,
                                base_model_prefix: str, skip_key: str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + ' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([
                module_name + "/" + sub_name
                for sub_name in encoder_modules.keys()
            ])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                            decoder_modules[decoder_name],
                            type(encoder_modules[encoder_name])) and len(
                                encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix,
                                       uninitialized_encoder_weights, skip_key)


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit,
               image_size,
               use_grad_checkpointing=False,
               ckpt_layer=0,
               drop_path_rate=0):

    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate)
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate)
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename,
                                           check_hash=False,
                                           progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
            state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg


def load_checkpoint_swinbase(model, url_or_filename, kwargs):
    if kwargs['image_size'] == 224:
        vision_config_path = 'Tag2Text/configs/swin/config_swinB_224.json'
    elif kwargs['image_size'] == 384:
        vision_config_path = 'Tag2Text/configs/swin/config_swinB_384.json'
    window_size = read_json(vision_config_path)['window_size']
    print('--------------')
    print(url_or_filename)
    print('--------------')
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename,
                                           check_hash=False,
                                           progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        if 'relative_position_bias_table' in k:
            dst_num_pos = (2 * window_size - 1)**2
            state_dict[k] = interpolate_relative_pos_embed(state_dict[k],
                                                           dst_num_pos,
                                                           param_name=k)
        elif ('relative_position_index' in k) or ('attn_mask' in k):
            del state_dict[k]
        elif "vision_multi" in k:
            state_dict[k.replace("vision_multi",
                                 "tagging_head")] = state_dict.pop(k)

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
