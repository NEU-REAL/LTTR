import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion_transformer import Encoder, Decoder

class RegionTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.build_transformer_module(model_cfg)
        self.similar_cfg = self.model_cfg.SIMILAIRY_COFIG
        self.build_similarity(self.similar_cfg)

    def build_transformer_module(self, config):
        transformer_cfg = config.TRANSFORMER_CONFIG
        if transformer_cfg.TRANS_MODE == 'Base':
            self.encoder_module = Encoder(
                image_size=transformer_cfg.IMAGE_SIZE, image_dim=transformer_cfg.IMAGE_DIM,
                patch_dim=transformer_cfg.PATCH_DIM, patch_size=transformer_cfg.PATCH_SIZE,
                pixel_dim=transformer_cfg.PIXEL_DIM, pixel_size=transformer_cfg.PIXEL_SIZE,
                depth=transformer_cfg.DEPTH, out_dim=transformer_cfg.OUT_DIM, 
                heads=transformer_cfg.HEADS, dim_head=transformer_cfg.DIM_HEADS
            )
            self.patch_trans = Decoder(
                dim=transformer_cfg.OUT_DIM, 
                dim_head=transformer_cfg.DIM_HEADS, 
                heads=transformer_cfg.HEADS,
                depth=transformer_cfg.DEPTH,
            )
        else:
            raise NotImplementedError
    
        self.to_patch = nn.Unfold(transformer_cfg.PATCH_SIZE, stride = transformer_cfg.PATCH_SIZE)
        self.to_image = nn.Fold(output_size=(transformer_cfg.IMAGE_SIZE,transformer_cfg.IMAGE_SIZE), 
                            kernel_size=transformer_cfg.PATCH_SIZE, stride=transformer_cfg.PATCH_SIZE)
        self.to_out = nn.Linear(transformer_cfg.OUT_DIM, 1)
        
    def encoder_decoder_forward(self, sf, tf):
        sf_encoder_feature, sf_pos_emb = self.encoder_module(sf)
        sf_patch_feature = sf_encoder_feature[:,1:,:]

        tf_encoder_feature, tf_pos_emb = self.encoder_module(tf)
        tf_patch_feature = tf_encoder_feature[:,1:,:]
        tf_patch = self.to_patch(tf)
        tf_patch_weight = self.to_out(tf_patch_feature).transpose(2,1)
        tf_patch_weight = tf_patch_weight.softmax(dim=2)
        tf_patch = tf_patch_weight * tf_patch
        tf_feature = self.to_image(tf_patch)

        tf_token_feature = tf_encoder_feature[:,1:,:]

        sf_decoder_feature = self.patch_trans(sf_patch_feature, tf_token_feature)
        sf_decoder_feature = self.to_out(sf_decoder_feature).transpose(2,1)
        sf_decoder_feature = sf_decoder_feature.softmax(dim=2)

        sf_patch = self.to_patch(sf)
        sf_trans_feature = sf_decoder_feature * sf_patch
        sf_trans_feature = self.to_image(sf_trans_feature)

        return sf_trans_feature, tf_feature

    def build_similarity(self, config):
        mode = config.MODE
        if mode == 'Base':
            self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)
            self.simi_func = self.base_similarity
            self.siam_out = nn.Conv2d(config.INPUT, config.OUTPUT, 1)
        elif mode == 'DepthWise':
            self.simi_func = self.xcorr_depthwise
            self.x_conv = nn.Sequential(
                nn.Conv2d(config.INPUT, config.OUTPUT, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.OUTPUT),
                nn.ReLU(inplace=True),
            )
            self.t_conv = nn.Sequential(
                nn.Conv2d(config.INPUT, config.OUTPUT, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.OUTPUT),
                nn.ReLU(inplace=True),
            )
        elif mode == 'Detect':
            self.simi_func = self.no_similarity
        else:
            raise NotImplementedError

    def no_similarity(self, sf, tf):
        return sf

    def base_similarity(self, sf, tf):
        siam = self.cos_layer(sf, tf).unsqueeze(1)
        fusion = sf * siam
        fusion = self.siam_out(fusion)
        return fusion

    def xcorr_depthwise(self, sf, tf):
        x = self.x_conv(sf)
        tf = self.t_conv(tf)
        batch = tf.shape[0]
        channel = tf.shape[1]
        x = x.view(1, batch*channel, x.shape[2], x.shape[3])
        tf = tf.view(batch*channel, 1, tf.shape[2], tf.shape[3])
        out = F.conv2d(x, tf, groups=batch*channel)
        out = out.view(batch, channel, out.shape[2], out.shape[3])

        return out * sf
