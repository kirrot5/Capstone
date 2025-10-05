import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='C:/Users/17890/Desktop/R2Gen/data/mimic_cxr/images')
    parser.add_argument('--ann_path', type=str,
                        default='C:/Users/17890/Desktop/R2Gen/data/mimic_cxr/eyegaze_annotation.json')
    parser.add_argument('--vocab_ann_path', type=str,
                        default=r"C:\Users\17890\Downloads\dataset\mimic_cxr_chen\annotation.json",
                        help='Path to the full annotation JSON file used ONLY for vocabulary building (to match pretrained model).')

    parser.add_argument('--pretrained_model', type=str,
                        default='C:/Users/17890/Desktop/R2Gen/pretrained/model_mimic_cxr.pth')
    parser.add_argument('--load_pretrained', type=int, default=1,
                        help='whether to load pretrained R2Gen model')

    # Eye-gaze settings
    parser.add_argument('--use_eye_gaze', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--fixation_path', type=str,
                        default='C:/Users/17890/Desktop/R2Gen/data/mimic_cxr/gaze/fixations.csv')
    parser.add_argument('--master_sheet_path', type=str,
                        default='C:/Users/17890/Desktop/R2Gen/data/mimic_cxr/gaze/master_sheet.csv',
                        help='path to master_sheet.csv')
    parser.add_argument('--max_fixations', type=int, default=50, help='maximum number of fixations')
    parser.add_argument('--gaze_encoder_type', type=str, default='lstm', choices=['lstm', 'gru'],
                        help='type of gaze encoder')
    parser.add_argument('--d_gaze', type=int, default=256, help='dimension of gaze features')
    parser.add_argument('--freeze_visual_extractor', type=int, default=1, help='freeze visual extractor to save memory')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.2, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    # ✅ 调整：语言侧扰动回到 0.5（更稳的泛化）
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='C:/Users/17890/Desktop/R2Gen/results/mimic_eyegaze',
                        help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='C:/Users/17890/Desktop/R2Gen/records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    # ✅ 调整：早停耐心缩短到 7（锁住中期最优）
    parser.add_argument('--early_stop', type=int, default=7, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=1e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=5e-5, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    # ✅ 调整：更强衰减，防止后期“越练越差”
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    # Label Smoothing & Grad Clip（保持不变）
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing factor for loss computation')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='gradient clipping threshold (clip_grad_norm)')

    args = parser.parse_args()
    return args



def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='validate', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # 加载预训练的 R2Gen 模型
    if args.load_pretrained and args.pretrained_model:
        print(f"\n{'=' * 50}")
        print(f"Loading pretrained R2Gen from:")
        print(f"  {args.pretrained_model}")

        try:
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')

            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            model_dict = model.state_dict()

            filtered_dict = {}
            skipped_keys = []

            for k, v in pretrained_dict.items():
                if 'gaze_encoder' in k or 'gaze_fusion' in k:
                    skipped_keys.append(k)
                    continue

                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(f"  Shape mismatch for {k}: "
                              f"pretrained {v.shape} vs model {model_dict[k].shape}")
                        skipped_keys.append(k)

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)

            print(f"Successfully loaded {len(filtered_dict)} layers")
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} layers (eye-gaze or mismatched)")
            print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print(f"  Training from scratch...")
            print(f"{'=' * 50}\n")

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                      train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()