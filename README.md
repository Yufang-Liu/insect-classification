# Insect classification task

# Requirements

+ anaconda3-2019.03 (python 3.7.3)
+ pytorch==1.1.0
+ configargparse==0.14.0
+ pydub==0.23.1
+ flask==1.0.2
+ sox
+ ffmpeg
+ lame

# Train and Evaluation
## Step 1: Preprocess the data
```bash
cd run
python preprocess.py -data_dir ../demo/data -train_path ../demo/train.txt -valid_path ../demo/valid.txt -save_data ../demo/demo
```

After running the preprocessing, the following files are generated in `demo` folder.

+ `demo.train.pt`: serialized pytorch file containing training data
+ `demo.valid.pt`: serialized pytorch file containing validation data
+ `demo.vocab.pkl`: serialized pickle file contraining vocabulary data

## Step 2: Train the model

```bash
python train.py -data ../demo/demo -save_model ../demo/demo-model
```

The main train command is quite simple. Minimally it takes a data file and a save file.

If you want to train on GPU, you need to set, as an example: CUDA_VISIBLE_DEVICES=1,3 `-gpu_ranks 0 1` to use
(say) GPU 1 and 3 on this node only.

the `demo-model_best.pt` file is generated in `demo` folder, which is the serialized pytorch file containing
model parameters, optimizer parameters, vocabulary, the score of validation set and the number of epoch.

## Step 3: Evaluation
```bash
python evaluation.py -model ../demo/demo-model_best.pt -data_dir ../demo/data  -test_path ../demo/valid.txt
```

Now you have a model which you can use to predict on test data. 

# Deployment
Now you have a model `demo-model_best.pt` in `demo` folder.

## Starting the pytorch server
```bash
cd deploy
python run_server.py -model ../demo/demo-model_best.pt
```

## Submitting requests to pytorch server
```bash
python simple_request.py -audio_path 测试04_黄脸油葫芦.m4a
```

You can see the following output at the bottom of screen.
```bash
1. 黄脸油葫芦: 0.6078
2. 迷卡斗蟋: 0.3922
```
