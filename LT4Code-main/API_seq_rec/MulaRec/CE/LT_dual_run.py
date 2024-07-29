from models.LT_dual_model import MulaRecDual
from utils import set_logger
from datetime import datetime
import pytz
import logging
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True,
                        help="max length")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="training batch size")
    parser.add_argument("--epoch", type=int, required=True,
                        help="number of epochs")
    parser.add_argument("--fuse", type=bool, default=False,
                        help="fuse or not")
    parser.add_argument("--norm", type=bool, default=False,
                        help="normalize or not")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory")
    parser.add_argument("--load_model_path", default=None,
                        help="load model from")
    parser.add_argument("--LT_solution", default='ce', type=str,
                        help="The LT solution.")
    parser.add_argument("--sample_per_class_file", default='../data/codebert_token_freq_file.pkl', type=str,
                        help="The file paths to the tokeb/class frequency.")
    parser.add_argument("--data_dir", type=str, required=True)

    args = parser.parse_args()
    os.makedirs('./log', exist_ok=True)
    set_logger('./log/dual_{}.log'.format(datetime.now(pytz.timezone('Asia/Singapore'))))
    logging.info(args)
    logging.info('Training Dual Model')

    # Load the fine-tuned model
    model = MulaRecDual(codebert_path='microsoft/codebert-base',
                        decoder_layers=6,
                        fix_encoder=False,
                        beam_size=5,
                        max_source_length=args.max_length,
                        max_target_length=args.max_length,
                        load_model_path=args.load_model_path,
                        l2_norm=args.norm,
                        fusion=args.fuse
                        )
     
 
    """
    import pickle
    if args.sample_per_class_file != None:
        f_r_ = open(args.sample_per_class_file, 'rb')
        samples_per_cls_vec = pickle.load(f_r_)
    """
    samples_per_cls_vec = None
    

    # train model
    model.train(
        train_filename=args.data_dir+'train_3_lines.csv',
        train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        learning_rate=5e-5,
        do_eval=True,
        dev_filename=args.data_dir+'validate_3_lines.csv',
        eval_batch_size=64,
        output_dir=args.output_dir,
        loss_flag=args.LT_solution,
        samples_per_cls=samples_per_cls_vec)
