**IMDB Sentiment Analysis using LSTM**:

Developed an end-to-end Natural Language Processing (NLP) and Deep Learning application for automated movie review sentiment classification using Python, TensorFlow, and Keras. The system leverages the IMDB dataset to classify reviews as positive or negative by learning contextual and sequential patterns in textual data.
- Created and configured a dedicated Python virtual environment with the required machine learning and deep learning dependencies.
- Preprocessed the IMDB dataset by restricting the vocabulary to the 10,000 most frequently occurring words and applying sequence padding to maintain a fixed input length of 200 tokens.
- Implemented an Embedding layer to transform encoded words into dense 32-dimensional vector representations.
- Designed an LSTM-based neural network with 64 hidden units to capture sequential dependencies and contextual information within movie reviews.
- Trained the model using the Adam optimizer and Binary Cross-Entropy loss, with validation data for performance monitoring.
- Evaluated the trained model on unseen test data and implemented real-time sentiment prediction with probability-based classification.
- Built a review decoding mechanism to convert numerical IMDB sequences back into human-readable text for model interpretation and validation.
Technology Stack: Python, TensorFlow, Keras, LSTM, NLP, Deep Learning, NumPy, IMDB Dataset, Virtual Environment
