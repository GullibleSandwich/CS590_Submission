CS590 Project Code Files
Added Files
1. DDPM/DDIM Notebook - https://www.kaggle.com/code/harshkatara/ddpm-cifar
    Instructions to run - Just run in the default kaggle runtime and may switch sampling methods as ddpm/ddim through functions

2. DDPM/DDIM Finetuning Notebook - https://colab.research.google.com/drive/1uhrhSlZm4DB8PRlClxvBXm_1Mxo8KmbC?usp=sharing
    Instructions to run - Just run the colab notebook as usual, You will need to generate a hugging face access token to access the datasets
    and the pretrained pipelines

    Imports and requirements are inbuilt to the notebook

3. EDM/ AOT improved Codebase - 
    Download pretrained checkpoints from https://drive.google.com/drive/folders/1AtjXcc1fineNJCSjuhNTQWU1z6TusyBw?usp=sharing
    Environment Setup - conda env create -f environment.yml
    Running for generating the 64X64 grid from pretrained models: python example.py
    Training the model - python train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp (Conditional)
                         python train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp (Unconditional) 
    Generating the images - python generate.py --network<Snapshot of model> --seeds 0-49999 --outdir<Outputdirectory> --subdirs --batch 200  --rho 90 --steps 14 