# DAGA2021_Conference

**Assessment and Evaluation of an Unsupervised Machine Learning Model for Automotive and Industrial NVH Applications**

**Abstract**

Rapid changes in the global industry like the emergence of electric vehicles and high-resolution data have posed new challenges for NVH engineers. Current analysis techniques involve an interdisciplinary knowledge of structural dynamics, signal processing and psychoacoustics but most notably they require experienced professionals to analyse and assess the ever-expanding amount of acquired industrial NVH data. Concurrently recent advances in machine learning show data driven model inference of feature representations- without human intervention. Unsupervised data driven methods have the potential to support NVH teams to focus on actual solutions by reducing manual efforts for pre-processing, classification and assessment of measurement and simulation-based data.

[conference_paper.pdf](https://github.com/tui-abdul/DAGA2021_Conference/files/7281816/daga_paper.pdf)

[conference_presentation_daga.pdf](https://github.com/tui-abdul/DAGA2021_Conference/files/7281815/Oral_presentation_daga_4.pdf)

Video Presentation
https://www.youtube.com/embed/3yEo0NBNgXc


## Tutorial: Training an Autoencoder with K-Fold Cross-Validation and MSE Analysis

This tutorial explains the workflow of training an autoencoder model for anomaly detection using TensorFlow/Keras, focusing on the following steps:
1. Loading and preprocessing the data.
2. Building the autoencoder model.
3. Training the model using K-Fold Cross-Validation.
4. Evaluating the model using Mean Squared Error (MSE) analysis.
5. Visualizing and saving results.

### Overview of the Key Components

1. **Autoencoder**: A type of neural network where the goal is to reconstruct input data by compressing it into a lower-dimensional space (encoding) and then expanding it back to the original space (decoding). It helps identify anomalies by evaluating how well the model can reconstruct the input.
  
2. **K-Fold Cross-Validation**: A model validation technique where the dataset is split into K parts (or folds). Each fold acts as a validation set once while the others serve as the training set. This helps to get a robust estimate of model performance by training and validating on multiple data splits.

3. **Mean Squared Error (MSE) Analysis**: MSE measures the average squared difference between the input data and its reconstructed version from the autoencoder. Anomalies are often detected when the reconstruction error (MSE) is significantly higher for faulty data than for normal data.

### 1. **Data Loading and Preprocessing**

The first step in training the model is loading the dataset from JSON files and preparing it for input into the model. The data consists of sequences (`order` and `mapping`), and is split into training and testing sets. The preprocessing involves reshaping the data into a 2D format (required for feeding into the dense layers) and normalizing the input features using MinMaxScaler to ensure they lie within a consistent range, typically [0, 1].

### 2. **Autoencoder Model Architecture**

The autoencoder model is built with fully connected layers (Dense layers). The network follows this structure:
- **Encoder**: A series of layers that compress the input data into a latent space (a lower-dimensional representation).
- **Latent Space**: A bottleneck layer that captures the most important information about the input data.
- **Decoder**: A mirrored series of layers that attempt to reconstruct the original data from the compressed latent representation.

This architecture allows the model to learn how to encode the essential patterns of the input data and then reconstruct it. If the model performs well on normal data but poorly on faulty data (with high reconstruction error), the model can be used for anomaly detection.

### 3. **Training with K-Fold Cross-Validation**

K-Fold Cross-Validation is used to split the dataset into multiple training and validation sets to improve the robustness of the model. Instead of training on a single train-validation split, K-Fold ensures that the model is validated on different parts of the dataset, giving a better indication of how well it will generalize.

During each fold:
- The model is trained on the training set.
- Validation is performed on the validation set.
- Early stopping is used to prevent overfitting, stopping training when validation performance stops improving.

### 4. **Model Evaluation and Saving Results**

After training, the model is evaluated by measuring how well it reconstructs the input data:
- **Training Reconstruction**: The model tries to reconstruct the training data as accurately as possible, and MSE is used to measure the reconstruction error.
- **Testing Reconstruction**: The model's performance is tested on unseen data, specifically faulty data. A higher MSE for faulty data compared to normal data indicates that the model can identify anomalies.

These reconstruction errors are stored in a DataFrame and saved into Excel files for further analysis.

### 5. **MSE Distribution and Visualizations**

After the model has been trained and evaluated, the reconstruction error (MSE) is visualized using histograms:
- **Training Set Distribution**: This shows how well the model reconstructs normal data. The MSE for normal data should ideally be low.
- **Test Set Distribution**: This represents the reconstruction error for faulty data, which is expected to have a higher MSE than the training data.

These visualizations help in understanding whether the model is successfully identifying anomalies. If the test set has consistently higher reconstruction errors than the training set, it indicates that the model can detect faulty or anomalous data.

### 6. **Saving the Model and Visualizing Training History**

The model is saved in TensorFlow format after training, making it reusable for inference or further training. The loss during training and validation is also visualized to track the model's learning process. The plotted loss curves give insight into whether the model is converging or overfitting.

### Conclusion

In this workflow:
- We loaded sequential data and preprocessed it for training.
- An autoencoder model was built and trained using K-Fold Cross-Validation for robust performance estimation.
- The reconstruction error (MSE) was used to identify faulty data.
- Results were saved, and visualizations of MSE distributions and loss curves were generated to evaluate the model's performance.

This process is effective for anomaly detection in sequential data, and it can be customized by adjusting the model architecture, training parameters, or evaluation metrics based on the specific use case.






This script demonstrates how to train an autoencoder with K-Fold Cross-Validation, visualize the reconstruction error, and detect anomalies using MSE distributions. You can customize the model architecture, learning rate, or dataset to adapt it to different tasks.
