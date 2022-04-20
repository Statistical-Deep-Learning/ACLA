### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

2. Specify '--dir_data' based on the HR and LR images path.

3. Organize training data like:
```bash
DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X1
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X1
```