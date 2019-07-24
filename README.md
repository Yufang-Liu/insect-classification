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

# Data Format
We will be working with some example data in `demo/` folder.
The directory structure is as follows:
<pre>
demo
└── all_data
    ├── 20190722
    │   ├── label.txt
    │   └── sound
    │       ├── 13_黄脸油葫芦.mp3
    │       ├── 测试01_迷卡斗蟋.m4a
    │       ├── 测试02_迷卡斗蟋.m4a
    │       ├── 测试03_迷卡斗蟋.m4a
    │       ├── 测试04_黄脸油葫芦.m4a
    │       ├── 测试05_黄脸油葫芦.m4a
    │       ├── 测试06_黄脸油葫芦.m4a
    │       └── 测试07_黄脸油葫芦.m4a
    └── 20190724
        ├── label.txt
        └── sound
</pre>
+ `all_data` folder  contains a lot of batch data.
Each batch of data corresponds to a folder.
This folder is recommended to be named by date, such as `20190722`.
If you want to add a batch of data, you should submit a folder like this. 
+ For each folder like `20190722`, it contains
    + `sound` folder  contains all audio files. Each audio file should be mp3, m4a, wav format and the name of file must end in format.
    + `label.txt` file annoates the label of audo files.
        + Each row is an anotation which includes two columns separated by a `\t`.
        + The first column is the name of audio file in `sound` folder.
        + The second column is the label of audio file.


# Train 
If you have prepared the data like `demo`, to train a model, you could run following order
```bash
./train demo
```

After runing the order, you could obtain the trained model named `demo-model_best.pt` in `demo` folder.

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
