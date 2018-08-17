# visualdialog-pytorch
Supervised and Multi-Agent Reinforcement Learning for Visual Dialog in Pytorch

This code is for our paper, <a href="https://arxiv.org/abs/1808.04359">Mind Your Language: Learning Visually Grounded Dialog in a Multi-Agent Setting</a>.

Work done with [Swami](https://github.com/swami1995) and [Vasu](https://vasusharma.github.io)

Can run multiple Q-Bots and multiple A-Bots, along with training from scratch (15 epochs of supervision), or load a pretrained model and do curriculum learning as defined in Das et al., or just do pure reinforcement learning via Reinforce. This can be set via flags defined in `arguments.py`

We used data from [VisDial](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/) but that has 512x7x7 image embeddings, in place of which we used 4096 size image embeddings from [VisDial v0.9](https://computing.ece.vt.edu/~abhshkdz/visdial/)

Script to evaluate mean rank, MRR, R@1, R@5 and R@10 coming soon

Credits to [Jiasen Lu](https://github.com/jiasenlu/visDial.pytorch) for his network definitions of the A-Bot encoders and decoders
