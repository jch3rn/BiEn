# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage


1. Download the processed data and put the data in  `../data/ACDC`

3. Train the model
```
cd code
python train_XXXXX_3D.py or python train_XXXXX_2D.py or bash train_acdc_XXXXX.sh
```

4. Test the model
```
python test_XXXXX.py
```
Our code is adapted from  [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks these authors for their efforts in building the research community in semi-supervised medical image segmentation.