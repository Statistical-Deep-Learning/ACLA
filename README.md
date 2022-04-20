# Adaptive Cross-Layer Attention for image restoration
## Train
#### Package Usage
torch >= 1.6.0
torch-summary

#### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.
```
|-- train
    |-- DIV2K
    |-- HR
        |-- 0001.png
        |-- 0002.png
        |-- ...
    |-- LR
        |-- X2
            |-- 0001x2.png
            |-- 0002x2.png
            |-- ...
        |-- X3
```


#### Benchmark Datasets
- Download benchmark datasets from http://vllab.ucmerced.edu/wlai24/LapSRN/

    - 2.1 Download and extract the ```SR_testing_datasets.zip```.
    - 2.2 Choose where you want to save the test dataset and put ```Prepare_TestData_HR_LR.m``` in this folder
    - 2.3 Specific the path_original in ```Prepare_TestData_HR_LR.m```
    - 2.4 Start the test dataset process with 
    ```
    matlab -nodesktop -nosplash -r Prepare_TestData_HR_LR
    ```

- final test dataset structure

```
|-- SR_testing_datasets.zip
|-- path_original         # <-- this is path_original
    |-- BSD100
    |-- Set5
    |-- Set14
    |-- Urban100
```

```
|-- your_test_data_file   #<-- will be used in model test
    |-- Prepare_TestData_HR_LR.m
    |-- HR
        |-- BSD100
        |-- Set5
        |-- Set14
        |-- Urban100
    |-- LR   
        |-- LRBI
            |-- BSD100
            |-- Set5
            |-- Set14
            |-- Urban100
```
#### Load the Model
1. Design your model in ```train/model```
2. Your model will be import from ```make_model``` function, so you must define this function in your custom model file.
3. If you have some custom setting in your model like the ```block_number```, you can transfer the parameters through ```--model_choose```.

```python
# A sample of make_model function
def make_model(args, parent=False):
    if args.model_choose == 'RCAN':   # default model
        return mymodel(scale=args.scale[0])
    elif args.model_choose.startswith('custom'): 
         # input like --model_choose custom_12_12_True 
        custom_args = args.model_choose.split('_')[1:]  
        # transfer True and False
        custom_args = [True if x=='True' for x in custom_args]
        custom_args = [False if x=='Flase' for x in custom_args]
        return mymodel(scale=args.scale[0], block_number1 = int(custom_args[0]), block_number2 = int(custom_args[1]), block_number3 = custom_args[2] )
```
#### Train the model
Training command example 
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model EDSR_DEFORM \
--save EDSR_deform_x2 \
--scale 2 \
--n_GPUs 1 \
--n_resblocks 16 \
--n_feats 64 \
--n_threads 4 \
--reset \
--chop \
--save_results \
--print_every 100 \
--batch_size 16 \
--test_every 1000 \
--ext bin \
--lr 1e-4 \
--lr_decay 200 \
--gamma 0.5 \
--epochs 1000 \
--loss 1*L1 \
--print_model \
--dir_data 'your data dir' \
--result_path 'your result path' \
--patch_size 96

```
