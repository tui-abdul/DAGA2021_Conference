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


In this workflow:
- We loaded sequential data and preprocessed it for training.
- An autoencoder model was built and trained using K-Fold Cross-Validation for robust performance estimation.
- The reconstruction error (MSE) was used to identify faulty data.
- Results were saved, and visualizations of MSE distributions and loss curves were generated to evaluate the model's performance.

This process is effective for anomaly detection in sequential data, and it can be customized by adjusting the model architecture, training parameters, or evaluation metrics based on the specific use case.






This script demonstrates how to train an autoencoder with K-Fold Cross-Validation, visualize the reconstruction error, and detect anomalies using MSE distributions. You can customize the model architecture, learning rate, or dataset to adapt it to different tasks.



## Tutorial: Saving STFT Results for Vibration Data in JSON Format

This tutorial walks through the process of extracting and saving Short-Time Fourier Transform (STFT) data for vibration analysis using MATLAB and Python. The code reads vibration data from `.mf4` files, extracts RPM-related slices, calculates order maps via MATLAB, and saves the results into JSON files for further analysis.

### 1. **Overview**

The code processes vibration data from multiple files, applying the following steps:
- **Data Loading**: Reads vibration and RPM data from `.mf4` files.
- **RPM Filtering**: Identifies and slices the portion of the data where the RPM is stable within a specific range.
- **STFT Calculation**: Uses MATLAB to compute order maps (STFT with RPM-based frequency scaling).
- **Saving Results**: Stores the computed STFT data and metadata in JSON format.

### 2. **Dependencies**

The following libraries and tools are required:
- **Python Packages**: `numpy`, `librosa`, `matplotlib`, `json`, and custom utility functions from `utility_functions`.
- **MATLAB Engine for Python**: Used to call MATLAB functions from Python. Make sure the `matlab.engine` package is installed and MATLAB is accessible.

### 3. **Functionality Breakdown**

#### 3.1. **Data Reading and RPM Filtering**
The function `save_stft()` takes in the root directory of the dataset and processes each file:
- **RPM Extraction**: The RPM channel is identified using the `mf4_reader_vib()` function. The RPM values are sliced to focus on a stable RPM range (e.g., 598-601 RPM). This ensures that only the relevant portion of the data is used.
  
- **Vibration Data Extraction**: Vibration data for each channel is extracted, synchronized with the corresponding RPM slice.

#### 3.2. **STFT Calculation Using MATLAB**
MATLAB is called through the `matlab.engine` interface to calculate the order map:
```python
mat = eng.rpmordermap(matlab.double(vib.tolist()), sample_rate, matlab.double(rpm.tolist()), 0.5, 'scale', 'db', 'Window', 'hann', 'amplitude', 'power');
```
- The MATLAB function `rpmordermap` computes the order map, where:
  - **vib**: Vibration data.
  - **rpm**: Corresponding RPM values.
  - **0.5**: Frequency resolution.
  - Other parameters control scaling, windowing, and amplitude calculation.

#### 3.3. **Data Padding**
To ensure consistency across samples, the computed STFT data (order map) is padded to a fixed size (`770x722`). The data is stored as a 2D array:
```python
mat_append = np.zeros((770, 722), dtype=np.float32)
mat_append[:,:len(mat[1,:])] = mat
```
This step ensures that all the STFT outputs have the same dimensions, even if some data segments are shorter.

#### 3.4. **Saving to JSON**
The processed data (order maps) and metadata (file names) are saved to JSON format:
```python
data['order'].append(mat_append.tolist())
data['mapping'].append(sample_name)
```
Each file's data is saved as a list of arrays, with a corresponding entry in the "mapping" list indicating the file it came from.

### 4. **How to Use the Code**

#### 4.1. **Set Paths**
Ensure that the paths to the datasets are correctly set:
```python
rootdirNormalData = "path to normal data"
jsonPathNormalData = "path to save normal data"
rootdirFaultData = "path to fault data"
jsonPathFaultData = "path to save fault data"
rootdirNormalDataFewSamples = "path to few sample normal data"
jsonPathNormalDataFewSample = "path to save few sample data"
```

#### 4.2. **Set Parameters**
You can modify the STFT parameters as needed. For example, change the FFT size (`n_fft`), hop length (`hop_length`), or the number of segments:
```python
save_stft(rootdirNormalData, jsonPathNormalData, n_fft=1024, hop_length=512, num_segments=10)
```

#### 4.3. **Run the Script**
After setting the paths and parameters, run the script to process the vibration data and save the results:
```python
save_stft(rootdirNormalData, jsonPathNormalData)
save_stft(rootdirFaultData, jsonPathFaultData)
save_stft(rootdirNormalDataFewSamples, jsonPathNormalDataFewSample)
```

### 5. **Output Format**

The output is saved as a JSON file with the following structure:
```json
{
  "mapping": ["file1.mf4", "file2.mf4", ...],
  "order": [
    [[stft_data_array_1], [stft_data_array_2], ...]
  ]
}
```
- **mapping**: Contains the filenames of the processed files.
- **order**: Contains the computed and padded STFT (order map) for each file.

### 6. **Extending the Code**

- **Custom Filters**: You can modify the RPM range or other filters applied to the data to fit your specific use case.
- **Feature Extraction**: The STFT output can be used for further feature extraction, such as detecting anomalies or trends.
- **Visualization**: You may extend the code to visualize the STFT outputs using `librosa` or `matplotlib` for easier data interpretation.

### Conclusion

This script provides a robust method for processing vibration data, computing STFT (order maps) via MATLAB, and saving the results in a structured JSON format. It's highly customizable, allowing you to modify parameters, filters, and output formatting according to your needs.
