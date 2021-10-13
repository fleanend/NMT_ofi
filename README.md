# Neural Machine Transliteration #
## Grafîa Ofiçiâ (Ligurian Language) to IPA ##

This project implements a simple transliterator to convert words from the Ligurian language (written in the [Grafîa Ofiçiâ](http://www.zeneize.net/grafia/index.htm)) to their pronunciation (written with IPA symbols). 

Under the hood the transliterator uses a character level Encoder-Decoder architecture with Attention.

### Install ###

Download the pretrained model from [here](https://www.dropbox.com/s/bhf4ubqbsq4fb09/ZE_IPA.zip?dl=1), unzip the contents in ./models

Create a virtual environment in yout preferred way and activate it.

Install requirements:

	pip install -r requirements.txt
	
### Sample Usage ###

From the repo root run:

	python bin/transliterate.py
	
### Data ###

You can check the data I used for training this model [here](https://www.kaggle.com/fleanend/ligurian-grafa-ofii-ipa).
	
### Acknowledgements ###

Thanks to [Bishal Santra](https://bsantraigi.github.io/), [whose code](https://bsantraigi.github.io/tutorial/2019/08/31/english-to-hindi-transliteration-using-seq2seq-model.html) and models I used as a basis for mine.