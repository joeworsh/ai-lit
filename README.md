# ai-lit - Deep Learning Tools for Computational Literature
This repository contains all of the models, infrastructure and data parsing tools to run Deep Learning experiments on the Project Gutenberg Literature Classification dataset.

# Table of Contents
1. [Installing Necessary Tools](#install)
2. [Gutenberg Dataset](#dataset)
3. [Running a Model](#model)

## Installing Necessary Tools <a name="install"></a>

TODO

## Gutenberg Dataset <a name="dataset"></a>

The Project Gutenberg dataset is hosted by the [Project Gutenberg](#https://www.gutenberg.org/wiki/Main_Page). The dataset can be downloaded by following the directions posted [here](#https://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages). The instructions for downloading the data used in the Genre Identification paper are as follows:
1. run ```wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=html&langs[]=en"``` in the dataset folder
2. download the XML/RDF catalog ```rdf-files.tar.zip``` [here](#https://www.gutenberg.org/wiki/Gutenberg:Feeds)
3. extract the XML/RDF library to the same directory as the downloaded dataset

After the dataset has been downloaded, the data must be built into a set of [TFRecord](#https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data) files which are read into the ai-lit models. Run any of the following python script to build the TFRecords for the different data representations.
* **TODO** run ```TODO```
* **TODO** run ```TODO```
* **TODO** run ```TODO```
* **TODO** run ```TODO```


## Running a Model <a name="model"></a>

TODO
