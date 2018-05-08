import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='/home/vd/visualdialog-pytorch/data/', help='path to directory with data')
  parser.add_argument('--input_img_h5', default='vdl_img_vgg.h5', help='path to dataset, now hdf5 file')
  parser.add_argument('--input_ques_h5', default='visdial_data.h5', help='path to dataset, now hdf5 file')
  parser.add_argument('--input_json', default='visdial_params.json', help='path to dataset, now hdf5 file')
  parser.add_argument('--outf', default='./save/train_allQ/', help='folder to output images and model checkpoints')
  parser.add_argument('--num_val', type=int, default=0, help='number of image split out as validation set.')
  parser.add_argument('--model_path', default='', help='path of saved network file for evaluation or continuing training')
  parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
  parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')
  parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
  parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')
  parser.add_argument('--workers', type=int, help='number of data loading workers', default=5)
  parser.add_argument('--batch_size', type=int, default=75, help='input batch size')
  parser.add_argument('--eval_iter', type=int, default=2, help='number of epochs after which we evaluate')
  parser.add_argument('--save_iter', type=int, default=1, help='number of epochs after which we save')
  parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is ADAM)')
  parser.add_argument('--Q_lr', type=float, default=1e-3, help='learning rate for QBot, default=1e-3')
  parser.add_argument('--A_lr', type=float, default=1e-3, help='learning rate for ABot, default=1e-3')
  parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
  parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
  parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
  parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')
  parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
  parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
  parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
  parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
  parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
  parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
  parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
  parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
  parser.add_argument('--image_loss_weight', type=float, default=1.0, help='folder to output images and model checkpoints')
  parser.add_argument('--gamma', type=float, default=0.99, help='discount')
  parser.add_argument('--eval', type=bool, default=False, help='Evaluation')
  parser.add_argument('--j_eval', type=bool, default=False, help='Jiasen Evaluation')
  parser.add_argument('--scratch', action='store_true', default=False, help='Train from Scratch')
  parser.add_argument('--curr', action='store_true', default=False, help='Train Entire Curriculum starting from SL pretraining')
  parser.add_argument('--start_curr', type=int, default=None, help='Value of K to start curriculum from')
  parser.add_argument('--k_curr', type=int, default=None, help='Do exactly one value of K always')  
  parser.add_argument('--num_abots', type=int, default=1, help='how many abots')
  parser.add_argument('--num_qbots', type=int, default=1, help='how many qbots')
  parser.add_argument('--img_feat_size', type=int, default=4096, help='size of image embedding')
  parser.add_argument('--no-save', action='store_true', default=False, help='specify if dont want to save model files')
  opt = parser.parse_args()

  return opt
