# DAGA2021_Conference

**Assessment and Evaluation of an Unsupervised Machine Learning Model for Automotive and Industrial NVH Applications**

**Abstract**

Rapid changes in the global industry like the emergence of electric vehicles and high-resolution data have posed new challenges for NVH engineers. Current analysis techniques involve an interdisciplinary knowledge of structural dynamics, signal processing and psychoacoustics but most notably they require experienced professionals to analyse and assess the ever-expanding amount of acquired industrial NVH data. Concurrently recent advances in machine learning show data driven model inference of feature representations- without human intervention. Unsupervised data driven methods have the potential to support NVH teams to focus on actual solutions by reducing manual efforts for pre-processing, classification and assessment of measurement and simulation-based data.

[conference_paper.pdf](https://github.com/tui-abdul/DAGA2021_Conference/files/7281816/daga_paper.pdf)

[conference_presentation_daga.pdf](https://github.com/tui-abdul/DAGA2021_Conference/files/7281815/Oral_presentation_daga_4.pdf)

Video Presentation
https://www.youtube.com/embed/3yEo0NBNgXc

# Tutorial: Training an Autoencoder Model with K-Fold Cross-Validation and MSE Analysis

This tutorial walks through a machine learning workflow using an autoencoder neural network built with TensorFlow and Keras. The autoencoder is trained on sequential data, with the aim of detecting anomalies or faulty patterns by reconstructing input data and calculating reconstruction errors. We'll explore the following steps:
- Loading and preprocessing data
- Building the autoencoder model
- Training with K-Fold Cross-Validation
- Evaluating the model using MSE (Mean Squared Error) on test and train sets
- Saving the results to files

### Key Components
1. **Autoencoder**: A type of neural network used to learn efficient representations (encoding) of input data and reconstruct it back.
2. **K-Fold Cross-Validation**: A technique that divides the dataset into multiple folds for robust model evaluation.
3. **MSE Analysis**: A way to detect anomalies by comparing reconstruction errors between the training and test sets.

### Requirements

Ensure you have the following installed:

```bash
pip install numpy tensorflow scikit-learn pandas librosa matplotlib
```

### 1. **Loading and Preprocessing Data**

We load data from JSON files and convert them into NumPy arrays for further processing. The data represents sequential information (`order` and `mapping`).

```python
def load_data(json_path):
    with open(json_path) as fp:
        data = json.load(fp)
    x = np.array(data["order"])
    z = np.array(data["mapping"])
    return x, z

# Similar functions for loading test data
```

- The function `load_data` reads the JSON files and returns the `order` and `mapping` as NumPy arrays.
- There are separate functions for training and test data, including different test sets to allow combining test data from multiple sources.

### 2. **Building the Autoencoder**

The `build_model` function constructs the autoencoder using Keras. The encoder learns compressed representations of the input, and the decoder reconstructs the input from this compressed form.

```python
def build_model(input_shape):
    input_data = layers.Input(input_shape)
    x = layers.Dense(256, activation='relu')(input_data)
    x = layers.Dense(128, activation='relu')(x)
    BN = layers.Dense(16, activation='relu', name='latent_space')(x)  # Latent space
    x = layers.Dense(128, activation='relu')(BN)
    output = layers.Dense(input_shape)(x)
    
    model = Model(input_data, output)  # Full autoencoder
    encoder = Model(input_data, BN)    # Encoder model
    return model, encoder
```

- **Latent Space**: A bottleneck layer that represents the compressed version of the input.
- **Model**: The autoencoder reconstructs the input, while the `encoder` is a separate model that only encodes the input data.

### 3. **Training the Model**

In the `train` function, the model is trained using K-Fold Cross-Validation. K-Fold splits the training data into multiple subsets, and training is done multiple times on different subsets.

```python
def train():
    x_train, y_train = load_data(json_path)
    x_test, y_test = load_data_test(json_path_faulty)
    
    x_train_reshaped = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
    model, encoder = build_model(input_shape=(x_train_reshaped.shape[1]))
    
    optimiser = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser, loss='mse', metrics=['mean_squared_error'])
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=5)
    for train_index, validation_index in kf.split(x_train_reshaped):
        x_train_fold = x_train_reshaped[train_index]
        x_val_fold = x_train_reshaped[validation_index]
        history = model.fit(x_train_fold, x_train_fold, validation_data=(x_val_fold, x_val_fold), epochs=50, batch_size=10, callbacks=[early_stop])
        break
```

- **KFold**: Divides the dataset into 5 folds and trains the model multiple times to evaluate performance across different splits.
- **Early Stopping**: The training is stopped early if the validation loss does not improve for a few epochs.

### 4. **Normalizing and Scaling**

Data is normalized using `MinMaxScaler`, which scales features to a given range (default [0,1]).

```python
scalar = MinMaxScaler()
x_train_transformed = scalar.fit_transform(x_train_reshaped)
x_test_transformed = scalar.transform(x_test)
```

### 5. **Evaluating and Saving the Results**

After training, the model's performance is evaluated by comparing the MSE of the training and test sets. The results are then saved to an Excel file and plotted.

```python
def train():
    ...
    mse_test = np.mean(np.abs(x_test_transformed - reconstructions), axis=1)
    mse_train = np.mean(np.abs(x_train_transformed - reconstructions_train), axis=1)
    
    # Saving results
    df = pd.DataFrame({'mse_train': mse_train.tolist(), 'y_train': y_train.tolist()})
    df2 = pd.DataFrame({'mse_test': mse_test.tolist(), 'y_test': y_test.tolist()})
    df.to_excel("mse_train.xlsx")
    df2.to_excel("mse_test.xlsx")
    
    # Plotting results
    fig, ax = plt.subplots()
    ax.hist(mse_test, bins=20, density=True, alpha=0.75, color="green", label="Test Set")
    ax.hist(mse_train, bins=20, density=True, alpha=0.75, color="red", label="Train Set")
    plt.title("MSE Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
```

- **MSE Calculation**: The mean squared error is computed for both the training and test sets to evaluate reconstruction performance.
- **Histograms**: The histograms of reconstruction errors for training and test sets provide a visual insight into the model's performance.

### 6. **Saving the Model**

The model is saved in TensorFlow format using `save_model` for future use.

```python
model.save(filepath)
```

### 7. **Visualizing Training History**

The `plot_history` function plots the training and validation loss curves over epochs to visualize how the model performs during training.

```python
def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()
```

### Running the Code

To run the code:
1. Ensure the JSON data files are in the correct location.
2. Run the script with:

```bash
python your_script.py
```

The script will:
- Train the model with K-Fold Cross-Validation.
- Save results, including MSE distributions and the trained model.
- Display loss curves and histograms of reconstruction errors.

### Conclusion

This script demonstrates how to train an autoencoder with K-Fold Cross-Validation, visualize the reconstruction error, and detect anomalies using MSE distributions. You can customize the model architecture, learning rate, or dataset to adapt it to different tasks.
