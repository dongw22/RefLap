### Create Environment
1. Create Conda Environment
```
conda create --name RefLap python=3.10
conda activate RefLap
```

2. Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm kornia
```


### Pre-trained Model
- [Pre-trained Model for NTIRE 2025 Reflection Removal Challenge](https://mcmasteru365-my.sharepoint.com/:u:/g/personal/dongw22_mcmaster_ca/EfCfBTlMiIhJhbQR36xZfFwB1q-o-Q8vQ7FAimhnib2GtQ?e=6savpU).

### Our Submission on Test Sever
- [Our Test Output](https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/EtKaYv63dxpLkma8trJ3rFEBSjL2df48qosXepsxt0yryA?e=Md0Z2J).

### Testing
Download above saved models and put it into the folder ./Enhancement/weights. To test the model, you need to specify the input image path (`args.input_dir`) and pre-trained model path (`args.weights`) in `./Enhancement/test.py`. Then run
```bash
cd Enhancement
python Enhancement/test.py 
```
You can check the output in `test-results-ntire25`.


### Contact
If you have any question, please feel free to contact us via dongw22@mcmaster.ca.






https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/EtKaYv63dxpLkma8trJ3rFEBSjL2df48qosXepsxt0yryA?e=Md0Z2J
