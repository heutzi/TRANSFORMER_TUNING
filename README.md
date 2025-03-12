# Transformer tuning pipeline

## Overview
This Python script was used to train Transformer models on a corpus annotated for emotion. The annotation scheme is structured around four dimensions established in the evaluative model of emotion developed by Loeser ([2019](#1)). See Noblet ([2025](#2)) for a description of the annotation process and some analysis.

## Installation

To get started, clone this repository and install the dependencies:

```bash
git clone https://github.com/heutzi/TRANSFORMER_TUNING.git
pip install -r requirements.txt
```

## Usage

### Data Preparation
Ensure your dataset is properly formatted following the examples provided in the folders *trains* and *test*s, in *corpora*.

### Training
Run the training script with the following command (replacing Alibaba-NLP/gte-multilingual-base with any compatible model):

```bash
python script_classifier_piped.py Alibaba-NLP/gte-multilingual-base
```

## References
<a id="1">Loeser, F. (2019).</a>
Modélisation probabiliste de l’influence des émotions sur l’acceptabilité des inno-vations [Université Grenoble Alpes (ComUE)]. https://www.theses.fr/2019GREAH003

<a id="1">Noblet J. (2025).</a>
Le bruit dans la mesure de la composante cognitive de l’émotion pour l’évaluation de l’acceptabilité des innovations. Corpus, 26. Publisher : Bases, corpus et langage - UMR 6039, DOI : <a href="https://journals.openedition.org/corpus/9267" target="_blank">10.4000/1364u.</a>

## License
This project is licensed under the MIT License.

## Author
Jonas Noblet
