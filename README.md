# Fashion Recommender System with Deep Learning

A content-based fashion recommender that finds visually similar products. It extracts deep visual features with a pre-trained **ResNet50** and retrieves the closest matches using **k-Nearest Neighbors**.

## How It Works

1. **Feature extraction:** each product image is resized to 224x224 and passed through a frozen ResNet50 (ImageNet weights) with global max pooling. This produces a 2048-dimensional feature vector per image.
2. **Normalization:** vectors are L2-normalized, so Euclidean distance ranks products the same way cosine similarity does.
3. **Similarity search:** a k-Nearest Neighbors model (k=5, brute force) finds the most similar products to a query image.
4. **Front-end:** the recommender is wrapped in a Streamlit app where a user picks or uploads an image and sees the recommendations.

No model training is needed. Transfer learning provides the visual understanding, and no labels are required.

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle, with 44,441 product images. The notebook downloads it automatically with `kagglehub`.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, NumPy, Streamlit, KaggleHub

## Files

- `FashionRS.ipynb`: feature extraction, nearest-neighbor search and recommendation demo

## How to Run

1. Install the dependencies:
   ```bash
   pip install tensorflow scikit-learn numpy streamlit kagglehub
   ```
2. Open `FashionRS.ipynb` and run the cells. The dataset is downloaded on the first run.
3. To use the Streamlit app, place the generated `Images_features.pkl` and `filenames.pkl` next to the app file and run:
   ```bash
   streamlit run app.py
   ```

## Limitations

- The notebook demo extracts features for a small subset (5 images) to validate the pipeline. Full-catalog recommendations require extracting all 44,441 images.
- Results are checked visually. There is no quantitative evaluation yet.
- Similarity is purely visual and ignores category, price, color preference and user behavior.

## Next Steps

- Extract features for the full dataset with batched predictions.
- Evaluate with precision@k using the dataset's category labels.
- Use an approximate nearest-neighbor index (for example FAISS) for faster search.
- Combine visual similarity with metadata filters such as category, gender and color.
