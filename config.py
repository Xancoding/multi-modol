import argparse


def get_config():
    parser = argparse.ArgumentParser(description='Open-Set Semi-Supervised Learning Framework')

    # data_prefix = './data'
    data_prefix = '/root/code/data'
    # args for path
    parser.add_argument('--data_dir', default=f'{data_prefix}/raw_data', help='data dir for cry_detection')
    parser.add_argument('--cross_val_dir', default=f'{data_prefix}/cross_validation/',
                        help='the splited data dir')
    parser.add_argument('--feature_dir', default=f'{data_prefix}/feature/',
                        help='feature dir')
    parser.add_argument('--model_save_dir', default='./checkpoints/',
                        help='the output dir for model checkpoints')
    parser.add_argument('--log_dir', default='./logs/',
                        help='log dir')
    # other args
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # pre_processing args
    parser.add_argument("--resize", type=int, nargs=2, default=[480, 300], 
                    help="Target size (width height), e.g., 480 300")
    parser.add_argument('--sample_rate', default=16000, type=int)
    parser.add_argument('--n_fft', default=256, type=int)
    parser.add_argument('--win_length', default=256, type=int)
    parser.add_argument('--hop_length', default=128, type=int)
    parser.add_argument('--n_mels', default=64, type=int)
    parser.add_argument('--clip_length', default=5, type=float)
    # parser.add_argument('--max_len', default=256, type=int)

    # 在get_config()函数中添加
    parser.add_argument('--mm_config', default='cam1', type=str, 
                   choices=['all', 'cam0', 'cam1'],
                   help='multi-modal configuration: '
                        'all=audio+cam0+cam1, '
                        'cam0=audio+cam0, '
                        'cam1=audio+cam1, ')
    parser.add_argument('--cuda', default=3, type=int)
    # train args
    parser.add_argument('--mode', default='mm', type=str, help='mm: multi-modal, audio')
    parser.add_argument('--audio_model', default='res18', type=str)
    parser.add_argument('--video_model', default='slowfast_50', type=str)
    parser.add_argument('--fusion_mode', default='late_fusion', type=str)
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--train_epochs', default=100, type=int, help='Max training epoch')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--dropout_prob', default=0.5, type=float, help='drop out probability')
    parser.add_argument('--detection_threshold', default=0.7, type=float, help='pseudo labeling threshold')

    parser.add_argument('--optim', default='AdamW', type=str, help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max grad clip')
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    parser.add_argument('--train_batch_size', default=1 * 2, type=int)
    parser.add_argument('--val_batch_size', default=2 * 2, type=int)
    # parser.add_argument('--train_batch_size', default=4 * 4, type=int)
    # parser.add_argument('--val_batch_size', default=8 * 4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=False, type=bool)

    # log args
    parser.add_argument('--wandb_api_key', default='35e924836a4db234887370d2536e07e323e97d2d', type=str)
    parser.add_argument('--wandb_mode', default='offline', type=str)

    # test args
    parser.add_argument('--resume', default='', type=str)

    # metrics args
    parser.add_argument('--average', default='macro', type=str, help='average type, macro、micro or weighted')

    return parser.parse_args()
