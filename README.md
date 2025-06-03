Each folders in the main directory contains python codes 
that compute computation time calculations of solving system 
of matrices using corresponding built-in functions and also using 
a Higher Order Jacobi Method proposed in the following preprint:

https://arxiv.org/abs/2505.16906

Each folder contained its own local conda environment at the time of code execution.
Each conda environment was created using the following commands:

# NumPy
conda create --prefix ./matrix_numpy python=3.11 numpy=2.1

conda activate ./matrix_numpy
python -c "import numpy; print(numpy.__version__)"
conda deactivate

# SciPy
conda create --prefix ./matrix_scipy python=3.11 scipy=1.14

conda activate ./matrix_scipy
python -c "import scipy; print(scipy.__version__)"
conda deactivate

# Pytorch
conda create --prefix ./matrix_pytorch python=3.11

conda activate ./matrix_pytorch
Visit PyTorch website to get the following command for your system configuration:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

conda deactivate
