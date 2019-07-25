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
An example workspace is in `demo` directory. It contains the training set (`all_data`)
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
            ├── ...
</pre>

For serving the incremental training, the `all_data` directory contains subdirectories (e.g., `20190722`, `20190724`),
each subdirectory is a batch of data (e.g., collected every one or two days).
- `sound` contains sould files (with format wav/mp3/m4a)
- `label.txt` contains annotations in column format 
  + Each row is annotation which has two columns separated by a `\t`.
  + The first column is the name of an audio file in `sound` folder.
  + The second column is the label of the audio file.


# Training 
To train a model, 
```bash
./train demo v1
```
where argument `demo` is the workspace above and `v1` is the prefix of the output model (e.g., version numbers)
If it sucesses, we will see the following information and the training process is started.
```
Building adam optimizer...
Building training data iterator...
train data size: 98
Building validation data iterator...
valid data size: 38
Starting training on CPU, could be very slow
Epoch: 0 Step: 1 Loss: 10.077764 Instances per Sec: 1.220545
Evaluating validation set...
  * Epoch: 0 Step: 1 P@1  : 0.500000 Instances per Sec: 143.867318
  * Epoch: 0 Step: 1 P@10 : 1.000000 Instances per Sec: 143.867318
  * Epoch: 0 Step: 1 P@50 : 1.000000 Instances per Sec: 143.867318
  * Epoch: 0 Step: 1 P@100: 1.000000 Instances per Sec: 143.867318
  * Epoch: 0 MAP  : 0.750000 Instances per Sec: 143.867318
  * Achieving best score on validation set...
Saving checkpoint ../demo/v1-model_best.pt
Epoch: 1 Step: 1 Loss: 0.683140 Instances per Sec: 1.680890
Evaluating validation set...
  * Epoch: 1 Step: 1 P@1  : 0.500000 Instances per Sec: 144.663466
  * Epoch: 1 Step: 1 P@10 : 1.000000 Instances per Sec: 144.663466
  * Epoch: 1 Step: 1 P@50 : 1.000000 Instances per Sec: 144.663466
  * Epoch: 1 Step: 1 P@100: 1.000000 Instances per Sec: 144.663466
  * Epoch: 1 MAP  : 0.750000 Instances per Sec: 144.663466
```
When it finishes, a model file `v1-model_best.pt` will be generated in `demo`.

# Deployment

## Starting the Server
Given a model `demo/v1-model_best.pt`, we can use `deploy/run_server.py` to deploy a REST API.
```bash
cd deploy
python run_server.py -model ../demo/v1-model_best.pt -version v1
```

It will start serving at http://127.0.0.1:5000/predict/v1

## Submitting Requests 

```bash
python simple_request.py -audio_path 测试04_黄脸油葫芦.m4a -version v1
```

It will output
```bash
1. 黄脸油葫芦: 0.6078
2. 迷卡斗蟋: 0.3922
```

## Full Options 
Full options of  `run_server.py`  
```
% python3 run_server.py -h
usage: run_server.py [-h] --model MODEL [--gpu GPU] [--host HOST]
                     [--port PORT] [--version VERSION]

run_server.py

optional arguments:
  -h, --help            show this help message and exit

Model:
  --model MODEL, -model MODEL
                        Path to model .pt file(s).
  --gpu GPU, -gpu GPU   Device to run on

SERVER:
  --host HOST, -host HOST
                        The host url
  --port PORT, -port PORT
                        The port
  --version VERSION, -version VERSION
                        The version of model
```

Full options of `simple_request.py`
```
% python3 simple_request.py -h
usage: simple_request.py [-h] --audio_path AUDIO_PATH
                         [--rest_api_url REST_API_URL] [--top_num TOP_NUM]
                         [--version VERSION]

simple_request.py

optional arguments:
  -h, --help            show this help message and exit

Data:
  --audio_path AUDIO_PATH, -audio_path AUDIO_PATH
                        Path to audio file

Server:
  --rest_api_url REST_API_URL, -rest_api_url REST_API_URL
                        Initialize the PyTorch REST API endpoint URL
  --top_num TOP_NUM, -top_num TOP_NUM
                        Return top predictions
  --version VERSION, -version VERSION
                        The version of model
```
