# Libtorch implementation of Neural Networks and other models

This project focuses on implementing and transforming machine learning classifiers from Python (PyTorch) to C++. It includes a simple neural network classifier trained on the MNIST dataset, as well as more advanced models like ConvNeXt and MobileViT, which were evaluated on the Oxford-IIIT Pet Dataset (Cats vs. Dogs). The aim is to replicate and optimize these models in C++ for improved performance and deployment flexibility. Libtorch API : https://pytorch.org/cppdocs/index.html 

## Features

- Implements a custom neural network classifier, ConvNext and MobileVit
- Uses PyTorch C++ API for model definition and training
- Supports both CPU and GPU training
- Includes data normalization and random sampling
- Saves model

## Prerequisites

- C++11 compatible compiler or higher 
- CMake (version 3.10 or higher)
- PyTorch C++ (LibTorch)
- Boost Filesystem

## Project Structure

```
├── CMakeLists.txt              # CMake build configuration
├── data                        # Dataset directory
│   ├── README.md               # Dataset documentation
│   ├── t10k-images-idx3-ubyte  # MNIST test images
│   ├── t10k-labels-idx1-ubyte  # MNIST test labels
│   ├── train-images-idx3-ubyte # MNIST training images
│   └── train-labels-idx1-ubyte # MNIST training labels
├── include                     # Header files
│   ├── Architecture.hpp        # Neural network architecture definitions
│   ├── Backbone.hpp            # Base model backbone implementations
│   ├── Model.hpp               # Model class declarations
│   ├── Trainer.hpp             # Training loop implementation header
│   ├── Trainer.tpp             # Template implementation for trainers
│   └── utils.hpp               # Utility functions header
├── model                       # Saved model directory
│   └── mnist_convnext.pth      # Example saved model
├── README.md                   # Project documentation (this file)
└── src                         # Source files
    ├── Architecture.cpp        # Architecture implementations
    ├── Backbone.cpp            # Backbone model implementations
    ├── main.cpp                # Main entry point
    ├── Model.cpp               # Model class implementations
    ├── python                  # Python folder containing the model to be tested
    └── utils.cpp  
```

## Building the Project

1. Clone the repository
2. Create a build directory:
    ```
    mkdir build && cd build
    cmake ..
    cmake --build . -j2
    ```

## Usage

Run the compiled executable with the following command:
``` ./build/mnist_classifier <mnist_data_path> <save_path>```


- `<mnist_data_path>`: Path to the MNIST dataset
- `<save_path>`: Path to save the trained model and checkpoints

## Model Architecture

### The classifier uses a simple feedforward neural network with the following structure:

- Input layer: 784 neurons (28x28 image flattened)
- Hidden layers: 512 -> 256 -> 128 -> 64 neurons
- Output layer: 10 neurons (one for each digit)
- Number of parameters : 

### ConvNext 
The ConvNext model was ""translated"" from python to C++ format as closely as possible: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py and has the following structure : 

#### ConvNext Backbone

Input: Images with configurable channel count (typically 3 for RGB)  
Stem: Conv2d layer (kernel=4, stride=4, no padding) for initial feature extraction  
Feature Progression: Five-stage feature extraction (typically [32, 64, 128, 256, 512])  
Stage Distribution: 3-3-9-3 block arrangement across stages (Stage 3 has the deepest structure)

#### ConvNext Classifier 

Backbone: ConvNext feature extractor
Pooling: AdaptiveAvgPool2d to 1x1 spatial dimensions
Classification Head:
- Fully connected layer: Linear(features → 512)
- LayerNorm 
- AvgPooling
- GelU + Dropout (0.3)
- Final linear layer: 512 → num_classes(1 for binary and 37 for different breeds)

### MobileViT





## Training

Hyperparameter for the simple neural network : 
- Batch size: 256
- Number of epochs: 10
- Learning rate: 0.0015
- Optimizer: Adam

Hyperparameter for the ConvNeXt for breed classification: 
- Batch size: 32
- Number of epochs: 25
- Learning: 1e-4
- Optimizer: AdamW 



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request


