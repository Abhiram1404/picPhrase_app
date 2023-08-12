Project Title: picPharse

Description:
I worked on a project with the goal of predicting captions for input images. The project utilized a dataset containing 8,000 images, each associated with 5 captions. The core idea was to combine features extracted from both the image and its corresponding captions, concatenating them to predict the next word in the caption sequence. For this purpose, a hybrid model was implemented, combining Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks for text analysis.

Key Points:

Objective: Predict captions for input images.

Dataset: Utilized a dataset comprising 8,000 images, each paired with 5 captions.

Model Architecture: Employed a hybrid approach involving a VGG16-based CNN for image feature extraction and an LSTM for text analysis.

Metric: Evaluated the model's performance using the BLEU Score, with BLEU-1 score at 0.544 and BLEU-2 score at 0.319.

Environment: Developed and executed the project on the Kaggle platform.

Libraries: Utilized essential libraries including numpy, matplotlib, Keras, TensorFlow, and NLTK for efficient implementation.

Neural Network: The project leveraged the VGG16 network for image feature extraction and a CNN-LSTM hybrid network for combining features to generate captions.

Download the dataset from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k

