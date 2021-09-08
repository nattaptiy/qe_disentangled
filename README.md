# Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation

---

We propose a method to distil language-agnostic meaning embedding using a multilingual sentence encoder.
By removing language-specific information from the original embedding, we retrieve an embedding that fully represents the meaning of the sentence.
The proposed method relies only on parallel corpora without any human annotations.
Our meaning embedding allows for efficient cross-lingual sentence similarity estimation using a simple cosine similarity calculation.

## Installation

---

1. Install requirements

    <code> $ pip install -r requirements.txt </code>
   
2. Prepare the training data and its embedding from the model of your choice.

3. Edit the model in <code> train_model.py </code> and run:

    <code> $ python train_model.py </code>

4. The result will be written in <code> result.csv </code>