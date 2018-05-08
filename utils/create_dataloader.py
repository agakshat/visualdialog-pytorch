import torch.utils.data
import utils.dataloader as dl

def create_dataloader(opt):
  opt.input_img_h5 = opt.data_dir +  opt.input_img_h5
  opt.input_ques_h5 = opt.data_dir + opt.input_ques_h5
  opt.input_json = opt.data_dir + opt.input_json

  dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                  input_json=opt.input_json, negative_sample = opt.negative_sample,
                  num_val = opt.num_val, data_split = 'train')
  dataset_val = None
  #dataset_val = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
  #                input_json=opt.input_json, negative_sample = opt.negative_sample,
  #                num_val = opt.num_val, data_split = 'test')

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=False, num_workers=int(opt.workers))
  dataloader_val = None
  #dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
  #                                         shuffle=False, num_workers=int(opt.workers))
  return dataset,dataset_val,dataloader,dataloader_val
