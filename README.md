# Insect Sound Classification 

# Requirements

+ anaconda3-2019.03 (python 3.7.3)
+ pytorch==1.1.0
+ configargparse==0.14.0
+ pydub==0.23.1
+ flask==1.0.2
+ sox
+ ffmpeg
+ lame

# Data Organization
An example workspace is in `demo` directory. It contains the training set (`all_data` directory)
and the learned model (after training). 

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

For serving the incremental training, the `all_data` directory contains subdirectories (e.g., `20190722`, `20190724`),
each subdirectory is a batch of data (e.g., collected every one or two days).
- `sound` contains sould files
- `label.txt` are annotations of the training data in column format 
  + Each row is an anotation which has two columns separated by a `\t`.
  + The first column is the name of audio file in `sound` folder.
  + The second column is the label of audio file.


# Train 
To train a model, 
```bash
./train demo v1
```
If it sucesses, you will see the following information and the training process is started.
```
training log...
```
When it finishes, the trained model named `v1-model_best.pt` in `demo` folder.

# Deployment

## Starting the server
```bash
cd deploy
python run_server.py -model ../demo/v1-model_best.pt -version v1
```

It will start a server at http://host:port/predict/v1. 

## Submitting requests 

```bash
python simple_request.py -audio_path 测试04_黄脸油葫芦.m4a -version v1
```

You can see the following output at the bottom of screen.
```bash
1. 黄脸油葫芦: 0.6078
2. 迷卡斗蟋: 0.3922
```

Other usages of `run_server.py`  and `simple_request.py` can be found with `--help`
```
usage info of run_server and simple_request
```
