### Prerequisites 
* Ubuntu 22.04
* Python = 3.8
* CUDA 11.8
* A4000 GPU
### 1, Generate new data
- We use the Gemini 2.0 Flash Preview Image Generation model to generate curved-line images for the training process.

- Download this file: https://drive.google.com/file/d/1q5VlBTrW3o9jODZWevOSdagcnAbVEIWW/view?usp=sharing

- Update the following variables:
  - GEMINI_API_KEY: The API key used to access and utilize Google’s models for free.
  - input_folder: The folder containing sample images to be used by the model for generating similar outputs.
  - output folder:  The folder containing output images of the model
  - PROMPT: A detailed text description specifying the desired output image.

- How to obtain the GEMINI_API_KEY:
  - Visit the following link: https://ai.google.dev/aistudio
  - Sign in using your Google account.
  - Once logged in, click on "Get API key" at the top right of the page.
  - Click "Create API key", then copy the generated key.
  - Replace the GEMINI_API_KEY variable in the code with the copied key.
  - Finally, run the script using the command: python gen_synthetic.py

### 2, Annotation & Format Prepare
- Tool used: Labelme
- You need to annotate the lane markings that appear in the image as follows:
- 
<p align="center">
  <img src="https://github.com/user-attachments/assets/3706c642-9448-4c2b-97ae-204743e430bf" alt="image">
</p>

- Rule: If there are two lane markings in the middle, annotate by drawing a line right in the center between the two markings, as shown in the image above.
- Annotate example: Open a directory that contains images → Choose Edit → Create lanestrip. Start draw points along the lane and save with class name you want. I will use “road_lane”
- After you annotate and save (Ctrl + S, or enable auto save at File – Save Automatically). The annotation file will appear in the same folder in .json.

<p align="center">
  <img src="https://github.com/user-attachments/assets/543d524e-627b-424c-b437-139557bf3c07" alt="image">
</p>

- Now you need to convert the format to a new format called Tusimple to make it more suitable for training.
- Data format look like this:
```Shell
{
      'raw_file': str. path_to_image.
      'lanes': list. A list of lanes. For each list of one lane, the elements are width values on image.
      'h_samples': list. A list of height values corresponding to the 'lanes', which means len(h_samples) == len(lanes[i])
}
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/432344c7-bb51-47d1-b923-c1da25044466" alt="image">
</p>

- -2 in lanes means on some h_sample, there is no existing lane marking. The first existing point in the first lane is (498, 290).
- For simplicity, you can use a script (labelme_to_tusimple.py in the repo) to perform the conversion.

### 3, Dataset folder structure
```Shell
SESSION/
|------ trainval/
|------ test/
|------ generate_seg_tusimple.py
|------ labelme_to_tusimple.py
```
- Please follow the steps below  to preprocess the data before training:

```Shell
## Run this script:
python labelme_to_tusimple.py --root trainval --out_file trainval
python labelme_to_tusimple.py --root test --out_file test
```
- You will get trainval.json in folder trainval and test.json in folder test. Move 2 file to main folder SESSION
```Shell
## Run this script:
python generate_seg_tusimple.py --root ./
```
- You will get a folder seg_label containing the segmentation masks.
- After running all the scripts, the dataset folder structure will be as follows:
```Shell
SESSION/
|------ seg_label
|	|-----trainval/
|	|-----test/
|	|-----list/
|------ trainval/
|------ test/
|------ generate_seg_tusimple.py
|------ labelme_to_tusimple.py
```

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
