# Community Regularization of Visually Grounded Dialog

[Akshat Agarwal](https://agakshat.github.io)\*, [Swaminathan Gurumurthy](https://github.com/swami1995)\*, [Vasu Sharma](https://vasusharma.github.io), [Mike Lewis](http://www.pitt.edu/~cmlewis/), [Katia Sycara](http://www.cs.cmu.edu/~sycara/)
Carnegie Mellon University, University of Pittsburgh

This repository contains a PyTorch implementation for our arXiv paper [1808.04359](https://arxiv.org/abs/1808.04359) on Community Regularization for Visually Grounded Dialog. The task requires goal-oriented exchange of information in natural language, however asking the agents to maximize information exchange while requiring them to adhere to the rules of human languages is an ill-posed optimization problem. Our solution, Community Regularization, involves each agent interacting with and learning from multiple agents, which results in more grammatically correct, relevant and coherent dialog without sacrificing information exchange. If you find this work useful, please cite our paper using the following BibTeX:

    @article{agarwal2018mind,
    title={Mind Your Language: Learning Visually Grounded Dialog in a Multi-Agent Setting},
    author={Agarwal, Akshat and Gurumurthy, Swaminathan and Sharma, Vasu and Sycara, Katia},
    journal={arXiv preprint arXiv:1808.04359},
    year={2018}
    }

## Installation

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
# however, these data files have 512x7x7 image embeddings, in place of which we used 4096 size image embeddings. we download that in another folder
mkdir v09
cd v09
wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v0.9/visdial_params.json
wget https://computing.ece.vt.edu/~abhshkdz/visdial/data/v0.9/data_img_vgg16_pool5.h5



```
1. Code for training 2 agents to exchange information about an image in natural language via supervised and Multi-Agent reinforcement learning (with curriculum)
2. Code for evaluating answer retrieval (mean rank, MRR and Recall@k)


Can run multiple Q-Bots and multiple A-Bots, along with training from scratch (15 epochs of supervision), or load a pretrained model and do curriculum learning as defined in Das et al., or just do pure reinforcement learning via Reinforce. This can be set via flags defined in `arguments.py`

We used data from [VisDial](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/) but that has 512x7x7 image embeddings, in place of which we used 4096 size image embeddings from [VisDial v0.9](https://computing.ece.vt.edu/~abhshkdz/visdial/)

Script to evaluate mean rank, MRR, R@1, R@5 and R@10 coming soon

Credits to [Jiasen Lu](https://github.com/jiasenlu/visDial.pytorch) for his network definitions of the A-Bot encoders and decoders
