# DREAM : Disentangled REpresentation for language-Agnostic Meaning
## Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation

---

We propose a method to distil language-agnostic meaning embedding using a multilingual sentence encoder.
By removing language-specific information from the original embedding, we retrieve an embedding that fully represents the meaning of the sentence.
The proposed method relies only on parallel corpora without any human annotations.
Our meaning embedding allows for efficient cross-lingual sentence similarity estimation using a simple cosine similarity calculation.

## Installation

---

1. Install requirements

   ```
   $ pip install -r requirements.txt 
   ```
   
2. Prepare the training data and its embedding from the model of your choice.

3. Edit the model in <code> train_model.py </code> and run:

    ```
   $ python train_model.py 
   ```

4. The result will be written in <code> result.csv </code>

Paper link : https://aclanthology.org/2021.emnlp-main.612/

BibTex For citation
```
@inproceedings{tiyajamorn-etal-2021-language,
    title = "Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation",
    author = "Tiyajamorn, Nattapong  and
      Kajiwara, Tomoyuki  and
      Arase, Yuki  and
      Onizuka, Makoto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.612",
    pages = "7764--7774",
    abstract = "We propose a method to distill a language-agnostic meaning embedding from a multilingual sentence encoder. By removing language-specific information from the original embedding, we retrieve an embedding that fully represents the sentence{'}s meaning. The proposed method relies only on parallel corpora without any human annotations. Our meaning embedding allows efficient cross-lingual sentence similarity estimation by simple cosine similarity calculation. Experimental results on both quality estimation of machine translation and cross-lingual semantic textual similarity tasks reveal that our method consistently outperforms the strong baselines using the original multilingual embedding. Our method consistently improves the performance of any pre-trained multilingual sentence encoder, even in low-resource language pairs where only tens of thousands of parallel sentence pairs are available.",
}
```
