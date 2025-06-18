### Prerequisites 
* Ubuntu 22.04
* Python = 3.8
* CUDA 11.8
* A4000 GPU

### TRAINING & ONNX Export
* The training process uses my modified version of the source code.
* You can clone the repository here: [https://github.com/Rykkura/CLR_Net.git](https://github.com/Rykkura/CLR_Net.git)
* I recommend using a Conda virtual environment for setting up the environment. You may use a different virtual environment, but I cannot guarantee that it will work properly.
#### Install conda virtual environment:
* Go to terminal and run this:
```Shell 
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

## After download run: 
bash Anaconda3-2024.10-1-Linux-x86_64.sh

## Finish the step in the setting and close the terminal
```
#### Clone the repository and create conda virtual environment:
```Shell
conda create -n clrnet python=3.8 -y
conda activate clrnet
git clone https://github.com/Rykkura/CLR_Net.git
cd CLR_Net
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install -r requirements.txt 
python setup.py build develop
```

* If everything was installed OK, please run this to start the training process:
```Shell
python main.py configs/clrnet/clr_resnet34_tusimple.py --gpu 0
```
* After training and got the model. The model will be in .pth format and be saved in savedir/ folder. For testing/inference test, run this script:
```Shell
python detect.py configs/clrnet/clr_resnet34_tusimple.py --img images\
          --load_from your_model.pth --savedir ./vis
```
* You will start the export to onnx process, follow this step bellow:
```Shell
git clone https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo.git
cp clr_head.py to your_path/CLR_Net/clrnet/models/heads/
mkdir your_path/CLR_Net/modules/ 
cp grid_sample.py to your_path/CLR_Net/modules/
cp torch2onnx.py to your_path/CLR_Net/
python torch2onnx.py configs/clrnet/clr_resnet34_tusimple.py  --load_from your_model.pth
```
* or you can follow this step from this repo to export onnx: [https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo](https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo)
