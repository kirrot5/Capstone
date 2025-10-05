import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.gaze_encoder import PaperStyleGazeEncoder, DualStreamFusion


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        # Eye-gaze components (Heatmap-based with temporal encoding)
        self.use_eye_gaze = args.use_eye_gaze
        if self.use_eye_gaze:
            self.gaze_encoder = PaperStyleGazeEncoder(args)
            self.gaze_fusion = DualStreamFusion(
                d_visual=2048,         # 视觉特征维度 (ResNet输出)
                d_gaze=args.d_model,   # 眼动特征维度 (512)
                d_output=2048          # 输出维度，保持与视觉特征一致
            )
            print("Heatmap-based eye-gaze module initialized (temporal)")

        # Freeze visual extractor if specified
        if args.freeze_visual_extractor:
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            print("Visual extractor frozen")

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', heatmaps=None, heatmap_masks=None):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        # Apply heatmap-based gaze guidance
        if self.use_eye_gaze and heatmaps is not None:
            gaze_feats = self.gaze_encoder(heatmaps)  # (B, d_model)
            att_feats = self.gaze_fusion(att_feats, gaze_feats)  # (B, R, 2048)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', heatmaps=None, heatmap_masks=None):
        att_feats, fc_feats = self.visual_extractor(images)

        # Apply heatmap-based gaze guidance
        if self.use_eye_gaze and heatmaps is not None:
            gaze_feats = self.gaze_encoder(heatmaps)  # (B, d_model)
            att_feats = self.gaze_fusion(att_feats, gaze_feats)  # (B, R, 2048)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output