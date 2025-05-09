# Libtorch implementation of Neural Networks and other models

This project explores the implementation and transformation of machine learning classifiers from Python (PyTorch) to C++ using LibTorch, the official PyTorch C++ API. The primary goal is to investigate the feasibility and performance of replicating PyTorch-trained models in a C++ environment.

The project includes:  
 - Basic Neural Network Classifier: A simple fully connected network trained on the MNIST dataset.

 - Convolutional Neural Networks (CNNs): Lightweight CNN models trained and evaluated on the Oxford-IIIT Pet Dataset (Cats vs. Dogs).

 - Advanced Architectures: Models such as ConvNeXt and MobileViT were created by translating and mirrored the actual implementation from python as much as possible, but only the ConvNeXt was trained on the Oxford PET Dataset for binary classification.  
    - ConvNeXt python implementation : https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py and https://tech.bertelsmann.com/en/blog/articles/convnext   
    - MobileViT python implementation : https://github.com/wilile26811249/MobileViT/blob/main/models/module.py 

For more details on LibTorch, refer to the official documentation: https://pytorch.org/cppdocs/index.html   
**Further new models will be added.** 
## Features

- Implements a custom neural network classifier,Convolutional Neural Networks, ConvNext and MobileVit
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
Run following command for the simple Linear model: ``` ./build/mnist dataset <mnist_data_path> <save_path>```   

- `dataset` : Balise necessary to indicate the Mnist dataset. This is to indicate we will use the mnist dataset. 
- `<mnist_data_path>`: Path to the MNIST dataset directory
- `<save_path>`: Path to save the trained model  

Run this command for the Oxford Pet Datset: ``` .build/mnist image <image_path> <label_path> <save_path> [--net=<cnn|convnext|mobilevit>] [--binary] ```

- `image` : Balise to indicate we are using Oxford Pet Dataset which needs the following image path and the label path. 
- `<image_path>` : Path to the image folder directory
- `<label_path>` : Path to the label file(e.g. the list.txt)
- The parameter `--net` allows to choose the model in question between **[cnn,convnext,mobilevit]** through `--net=cnn` 
- The flag `--binary` just needs to be added to indicate if we are doing binary or multi class classfication. 

** RIGHT NOW, Only the ConvNeXt and CNN are used and trained on the Oxford Pet Dataset**

## Model Architecture

### The classifier uses a simple feedforward neural network with the following structure:

- Input layer: 784 neurons (28x28 image flattened)
- Hidden layers: 512 -> 256 -> 128 -> 64 neurons
- Output layer: 10 neurons (one for each digit)
- Number of parameters : 575050

### CNN + classifier
Created a simple CNN classifier with the following structure : 
#### CNN Backbone

- Input: Images with 3 channels (RGB)  
- Feature Progression: Five sequential convolutional blocks with increasing channels:
→ [64, 128, 256, 512, 1024] (or configurable via constructor)  
- Dropout (0.25)
- Block Structure, where each block contains:
    - Conv2d (kernel=3, stride=1, padding=1)
    - BatchNorm2d
    - ReLU

#### CNN Classifier

- Backbone: Custom CNN feature extractor (five blocks)  
- Pooling: AdaptiveAvgPool2d (output size = 7×7 spatial dimensions)
- Flattening: Tensor reshaped to (batch_size, features)
- Classification Head:
    - Linear layer: final layer output * 7 * 7 → 512
    - BatchNorm1d
    - ReLU
    - Dropout (0.5)
    - Final Linear: 512 → num_classes (1 for binary or any class count)

- Number of parameters : 15986241 for binary and 16004709 for breed classification


### ConvNext 
The ConvNext model was ""translated"" from python to C++ format as closely as possible: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py and has the following structure : 

#### ConvNext Backbone

- Input: Images with configurable channel count (typically 3 for RGB)  
- Stem: Conv2d layer (kernel=4, stride=4, no padding) for initial feature extraction  
- Feature Progression: Five-stage feature extraction (typically [32, 64, 128, 256, 512])  
- Stage Distribution: 3-3-9-3 block arrangement across stages (Stage 3 has the deepest structure)

#### ConvNext Classifier 

- Backbone: ConvNext feature extractor 
- Pooling: AdaptiveAvgPool2d to 1x1 spatial dimensions    
- Classification Head:  
    - Fully connected layer: Linear(features → 512)  
    - LayerNorm   
    - AvgPooling  
    - GelU + Dropout (0.3)  
    - Final linear layer: 512 → num_classes(1 for binary and 37 for different breeds)  

- Number of parameters: ~ 30 millions of the ConvNeXt Tiny and 13 millions parameter for a basic model. 
### MobileViT

#### MobileViT Backbone
- Input: Images with configurable channel count (typically 3 for RGB)
- Stem: Conv2d layer (kernel=3, stride=2, padding=1) followed by an InvertedResidual block
- Feature Progression: Five-stage hierarchical structure with progressive feature widths 
- Transformer Integration: MobileViT Blocks embedded in Stages 2, 3, and 4
- Stage Distribution:
    - Stage 1: 3 InvertedResidual blocks
    - Stage 2: 1 InvertedResidual + 1 MobileViT block
    - Stage 3: 1 InvertedResidual + 1 MobileViT block
    - Stage 4: 1 InvertedResidual + 1 MobileViT block + final 1×1 Conv2d layer

#### MobileViT Classifier

- Backbone: MobileViT feature extractor (as defined above)
- Pooling: AdaptiveAvgPool2d to reduce spatial dimensions to 1x1
- Classification Head:
    - Linear(features_list.back() → 512)
    - BatchNorm1d(512)
    - ReLU
    - Dropout(p=0.25)
    - Linear(512 → num_classes)

## Training

1. Hyperparameter for the simple neural network : 
    - Batch size: 256
    - Number of epochs: 10
    - Learning rate: 0.0015
    - Optimizer: Adam

    Results:
    **Avg Precision: 0.976797 | Avg Recall: 0.976623 | Avg F1 Score: 0.975966**  
    **Final training loss:** 0.045639
    **Final training accuracy:** 98.68
    **Final test loss:** 0.0720981
    **Final test accuracy:** 98.2  
    **Time taken:** 11390ms 


2. Hyperparameter for the Simple CNN model for binary classification: 
    - Batch size: 32
    - Number of epochs: 25
    - Learning: 1e-3
    - Optimizer: Adam

    Results: 
    **Avg Precision: 0.669067 | Avg Recall: 0.716161 | Avg F1 Score: 0.672006**  
    **Final training loss:** 0.400945
    **Final training accuracy:** 85.051
    **Final test loss:** 0.428794
    **Final test accuracy:** 84.4792  
    **Time taken:** 968926 ms

3. Hyperparameter for the ConvNeXt model for breed classification(used the ConvNeXt Tiny):
    - Batch size: 32
    - Number of epochs: 25
    - Learning: 1e-3
    - Optimizer: AdamW

    Results: 
    **Avg Precision: 0.243695 | Avg Recall: 0.119729 | Avg F1 Score: 0.135847**  
    **Final training loss:** 0.790147
    **Final training accuracy:** 63.6905
    **Final test loss:** 0.764939
    **Final test accuracy:** 66.9843  
    **Time taken:** 2601337 ms


## Observation

- I tried my best to translate from PyTorch to LibTorch but perhaps as you can see on the training with the ConvNext. 
- During the translation especially for the simple Neural Networks such as the Linear classifier and the CNN models, there didn't seem to be any problems to "translate" but 
for ConvNeXt the drop path method had to be implemented based on this : https://github.com/FrancescoSaverioZuppichini/DropPath  
- Another observation is that some aspects of PyTorch are lacking like the schedulers such as CosineAnnealingLr which is not present in the API but also this lack of documentation. 
- Another good aspect would be that the possibility of going from a model trained on PyTorch to C++ using torchscript : https://docs.pytorch.org/tutorials/advanced/cpp_export.html and https://zachcolinwolpe.medium.com/deploying-pytorch-models-in-c-79f4c80640be 
- Another observation that can be made would be to compare the training time using PyTorch and LibTorch which is not done here. 
- Currently it's not possible to go from LibTorch to PyTorch thus saving a model in LibTorch doesn't allow to load in PyTorch. 
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request
