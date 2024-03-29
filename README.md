# Likert Scoring with Grade Decoupling for Long-term Action Assessment

This is the code for CVPR2022 paper "Likert Scoring with Grade Decoupling for Long-term Action Assessment".

## Environments

- RTX2080Ti
- CUDA: 10.2
- Python: 3.9.7
- PyTorch: 1.10.1+cu102

## Features

The features and label files of Rhythmic Gymnastics dataset can be download [here](https://1drv.ms/u/s!AqXkt0Mw7p9llVaV2oV1mwmdAICG).

\[23-04-10 Update\] The features and label files of Fis-V dataset can be download [here](https://1drv.ms/u/s!AqXkt0Mw7p9llWEihc533CB87U5P?e=EadhCo).

## Running

Please fill in or select the args enclosed by {} first.

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --n_decoder 2 --n_query 4 --dropout 0.3 --test --ckpt {the name of the used checkpoint}
```

