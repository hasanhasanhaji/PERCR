# End-to-End Persian Coreference Resolution

This repository contains an implementation of an end-to-end coreference resolution system for Persian text. Coreference resolution is the task of identifying which mentions in a text refer to the same real-world entity. For example, in the sentence "Barack Obama gave a speech. He was very eloquent.", coreference resolution would link "Barack Obama" and "He."

**Based on:**

This code is based on the implementation found in the following repository:

* [https://github.com/shayneobrien/coreference-resolution](https://github.com/shayneobrien/coreference-resolution)

**Key Features**

* **End-to-End Model:**  The system employs a neural network model to directly predict coreference clusters in a document.
* **Bidirectional LSTM Encoder:** Utilizes a bidirectional LSTM to capture contextual information in the text.
* **GLoVe and Word2Vec Embeddings:** Leverages pre-trained word embeddings from both GLoVe and Word2Vec for representing words.
* **Character-Level CNN:**  Incorporates a character-level CNN to capture morphological information about words.
* **Mention Scoring:**  Scores individual spans (potential mentions) based on their likelihood of being coreferent.
* **Pairwise Scoring:**  Scores pairs of spans to determine if they refer to the same entity.
* **Pruning:**  Prunes the number of candidate mentions to improve efficiency.
* **Evaluation:**  Provides an evaluation script to calculate coreference resolution performance metrics.

**How to Run**

**Prerequisites:**

* **Python (3.6 or later):**  Ensure you have a compatible Python version installed.
* **PyTorch:** Install PyTorch following the instructions on the official website: [https://pytorch.org/](https://pytorch.org/)
* **Other Dependencies:**  Install the required Python packages:

   ```bash
   pip install -r requirements.txt
