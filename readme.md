### NeuralOMT - brain surface code

### Project Overview

This repository contains code for a 
"Revealing Cortical Spreading Pathway of Neuropathological Events by Neural Optimal Mass Transport". The codebase includes modules for dataset construction, loss functions, models, and utilities.

### Directory Structure

- **configs**: Configuration files for training and testing.
- **datasets**: Modules to construct datasets from various file formats such as vtk, csv, and gii.
- **loss**: Implementation of Wasserstein distance loss functions and ADMAE.
- **models**: Contains preprocessing functions for vtk features, feature extractors, and new model architectures.
- **utils**: Utility functions.

### Dataset Modules

- `surface_dataset.py`: Module to extract data from MGH and GII files.
- `surface_dataset_ori.py`: Module to extract data from MGH and GII files with additional parameter for left/right brain orientation.
- `surface_dataset_ADNI.py`: Module to extract data from TXT and GII files.
- `surface_dataset_ADNI_tau.py`: Module to extract data from CSV and GII files.
- `surface_dataset_ADNI_tau_ori.py`: Module to extract data from CSV and GII files with additional parameter for left/right brain orientation.
- `surface_dataset_ADNI_tau_vtk.py`: Module to extract data from CSV and VTK files.
- `surface_dataset_ADNI_tau_vtk_ori.py`: Module to extract data from CSV and VTK files with additional parameter for left/right brain orientation.

### Configuration Files

- `admaeloss_train_tau_163842_2gpu.yaml`
- `admaeloss2_train_tau_163842_2gpu.yaml`
- `admaeloss3_train_tau_163842_2gpu.yaml`
- `base_param.yaml`: Configuration for Mean Squared Error (MSE) training.
- `base_param_test.yaml`: Configuration for testing.
- `new_model_train_tau_163842_2gpu.yaml`: Configuration for training with MSE loss on two GPUs.
- `new_model_train_tau_163842_2gpu_resume.yaml`: Configuration for resuming training with MSE loss on two GPUs.
- `new_model_train_tau_163842_4gpu.yaml`: Configuration for training with MSE loss on four GPUs.
- `new_model_train_tau_163842_8gpu.yaml`: Configuration for training with MSE loss on eight GPUs.

### Important Parameters in Configs

- `input_dir`: Absolute path to the training dataset.
- `val_dir`: Absolute path to the validation dataset.
- `test_dir`: Absolute path to the test dataset.
- `output_dir`: Directory to store weights and results.
- `model_PE_dir`: File path to the DNN module (`DNN.py`).
- `model_PE`: Class name within `DNN.py`.
- `model_PE_params`: Parameters for the `DNN` class.
  - `infeature`: Input features.
  - `numclass`: Number of classes.

### Execution Commands

#### Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m torch.distributed.launch --nproc_per_node 8 --use_env Brain_surface_GAN_wdloss.py --config configs/new_model_train_tau_163842_8gpu.yaml > my_log_tau168342.log 2>&1 &
```

#### Testing Commands

```bash
python test_batch_disease.py --config configs/base_param_test.yaml
```
Calculate Mean Absolute Error (MAE) for each disease.

```bash
python test_single_disease.py --config configs/base_param_test.yaml
```
Calculate flow for each disease and save the results.

### Conclusion

This README provides an overview of the project structure, dataset modules, configuration files, and execution commands for training and testing. For detailed usage and further information, please refer to the documentation within each module or consult the project's documentation.
