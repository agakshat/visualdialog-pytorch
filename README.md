# Community Regularization of Visually Grounded Dialog

[Akshat Agarwal](https://agakshat.github.io)\*, [Swaminathan Gurumurthy](https://github.com/swami1995)\*, [Vasu Sharma](https://vasusharma.github.io), [Mike Lewis](http://www.pitt.edu/~cmlewis/), [Katia Sycara](http://www.cs.cmu.edu/~sycara/)

Carnegie Mellon University, University of Pittsburgh

This repository contains a PyTorch implementation for our arXiv paper [1808.04359](https://arxiv.org/abs/1808.04359) on Community Regularization for Visually Grounded Dialog. The task requires goal-oriented exchange of information in natural language, however asking the agents to maximize information exchange while requiring them to adhere to the rules of human languages is an ill-posed optimization problem. Our solution, Community Regularization, involves each agent interacting with and learning from multiple agents, which results in more grammatically correct, relevant and coherent dialog without sacrificing information exchange. If you find this work useful, please cite our paper using the following BibTeX:

    @article{agarwal2018community,
    title={Community Regularization of Visually-Grounded Dialog},
    author={Agarwal, Akshat and Gurumurthy, Swaminathan and Sharma, Vasu and Lewis, Mike and Sycara, Katia},
    journal={arXiv preprint arXiv:1808.04359},
    year={2018}
    }

## Installation and Downloading Data

```bash
# set up a clean virtual environment
virtualenv -p python3 ~/visualdialog
source ~/visualdialog/bin/activate # you will have to run this command in every new terminal, alternatively add macro to your .bashrc

pip3 install torch torchvision (or as appropriate from pytorch.org)
sudo apt-get install -y tensorboardX h5py 

git clone https://github.com/agakshat/visualdialog-pytorch.git
cd visualdialog-pytorch

# download visual dialog data
mkdir data
cd data
wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/vdl_img_vgg.h5
wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/visdial_data.h5
wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/visdial_params.json
# however, these data files have 512x7x7 image embeddings, in place of which we 
# used 4096 size image embeddings. we download that in another folder
mkdir v09
cd v09
wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v0.9/visdial_params.json
wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v0.9/data_img_vgg16_pool5.h5

mkdir save
```

## Training

```bash
# now run the code

# Option 1: Train from scratch, including 15 epochs of supervised learning
# followed by RL through curriculum
python main.py --num_abots 3 --num_qbots 1 --scratch --outf save/temp_dir

# Option 2: Start training from RL, assuming pretrained supervised learning agents
python main.py --num_abots 3 --num_qbots 1 --curr  --model_path save/pretrained_SL.pth --outf save/temp_dir
```
Important Command Line Arguments:
1. `--data_dir` specifies path to data folder. Default is `data/`
2. `--v09_data_dir` specifies path to alternative (v09 img files) data folder. Default is `data/v09/`
(There is no need to change these if you installed using the exact commands as above)
3. `--num_qbots` and `--num_abots` specifies number of Q-Bots and A-Bots, respectively
4. `--model_path` specifies the torch `.pt` file with the pretrained agents to be loaded. 
5. `--outf` specifies the save directory where the trained models will be saved epoch-wise, along with tensorboard logs
6. `--scratch` if specified, the agents are trained from scratch, starting with supervised learning
7. `--curr` if specified, the agents are trained from the beginning of the curriculum, assuming that `--model_path` has been specified to load SL pretrained model files
8. `--start_curr K` if specified, the agents start curriculum training not from the beginning, but after the first 10-K rounds of curriculum have happened. Look at `main.py` for details.
9. `--batch_size` default is 75, which you might need to reduce depending on the GPU being used. Note that as curriculum training progresses, progressively greater amount of GPU memory is used, becoming constant only when the agents are training purely via RL.

## Evaluation
```bash
# To run only the evaluation, get image retrieval percentile scores and/or view generated dialog:
python main.py --num_abots 3 --num_qbots 1 --curr  --model_path save/pretrained_SL.pth --outf save/temp_dir --eval 1

# To get answer retrieval Mean Rank, MRR and Recall@k metrics:
python evaluate_mrr.py --num_abots 3 --num_qbots 1 --model_path save/pretrained_model_file.pth
```

## Example of generated dialog
![ex](ex.png)

### Acknowledgement
Credits to [Jiasen Lu](https://github.com/jiasenlu/visDial.pytorch) for his network definitions of the A-Bot encoders and decoders
