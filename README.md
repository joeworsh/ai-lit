# ai-lit - Deep Learning Tools for Computational Literature
This repository contains all of the models, infrastructure and data parsing tools to run Deep Learning experiments on the Project Gutenberg Literature Classification dataset.

# Table of Contents
1. [Installing Necessary Tools](#install)
2. [Gutenberg Dataset](#dataset)
3. [Running a Model](#model)

## Installing Necessary Tools<a name="install"/>

### Python
ai-lit is implemented using python 3.5.3

Below are the needed python packages to run ai-lit. These can be installed through ```pip``` and it is best to be installed in a [virtual environment](https://docs.python.org/3.5/tutorial/venv.html).

* **nltk** 3.1

* **gensim** 1.0.1

* **glob2** 0.5

* **matplotlib** 1.5.1

* **numpy** 1.14.2

* **tensorflow** or **tensorflow-gpu** 1.3.0

### R
ai-lit also has modules written in R 3.3.2

Below are the needed R libraries to run ai-lit R models. These can be installed through the R package manager.

* **caret** 3.3.3

* **class** 3.3.2

* **dplyr** 3.3.2

* **jsonlite** 3.3.3

* **magrittr** 3.3.2

* **naivebayes** 3.3.3

* **plyr** 3.3.2

* **randomForest** 3.3.3

* **tm** 3.3.2

* **xgboost** 3.3.3

## Gutenberg Dataset<a name="dataset"/>

The Project Gutenberg dataset is hosted by the [Project Gutenberg](https://www.gutenberg.org/wiki/Main_Page). The dataset can be downloaded by following the directions posted [here](https://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages). The instructions for downloading the data used in the Genre Identification paper are as follows:

1. run ```wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"``` in the dataset folder. Note: this download can take two days and requires ~11GB of storage space.

2. download the XML/RDF catalog ```rdf-files.tar.zip``` [here](https://www.gutenberg.org/wiki/Gutenberg:Feeds).

3. extract the XML/RDF library to the same directory as the downloaded dataset

The downloading process should yeild a folder structure like the following:
|->GutenbergDataset
|-->cache
|-->www.gutenberg.org
|--><mirror name of download mirror>.
The name of the mirror does not matter, the data will be extracted from any mirror folder name.

After the dataset has been downloaded, the data must be built into a common format and then compiled into [TFRecord](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) files which are read into the ai-lit TensorFlow models. Run the following python script to build the common dataset format and the TFRecords for the different data representations.

* run ```python3 build_dataset.py <folder of the Gutenberg dataset>```


## Running a Model<a name="model"/>

TODO
