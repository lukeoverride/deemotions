# Deemotions

This repository contains the implementation of the method illustrated in:

[**Emotion Recognition in the Wild using Deep Neural Networks and Bayesian Classifiers**](https://arxiv.org/abs/1709.03820) 


## Preamble

This repository contains the code to perform emotion recognition both for the following datasets:
1) Emotion Recognition based on [CK+ dataset](https://ieeexplore.ieee.org/document/5543262/)
2) GAF (Group Affect Dataset) associated with [EmotiW Challenge 2017](https://sites.google.com/site/emotiwchallenge/)
and for the GAF dataset. 

The source files for this work are contained _into the "wild"_ folder (it is so funny, isn't it?).
This method has been published in the paper "Emotion Recognition in the Wild Using Deep Neural Networks and Bayesian Classifiers". Please cite us if you are using our code, or if you are finding it useful for your research.

Cite:

`
@inproceedings{surace2017emotion,
  title={Emotion recognition in the wild using deep neural networks and Bayesian classifiers},
  author={Surace, Luca and Patacchiola, Massimiliano and Battini S{\"o}nmez, Elena and Spataro, William and Cangelosi, Angelo},
  booktitle={Proceedings of the 19th ACM International Conference on Multimodal Interaction},
  pages={593--597},
  year={2017},
  organization={ACM}
}
`

**The rest of the readme is referring to the EmotiW code**

## Contributors

[Luca Surace](https://cassiophong.wordpress.com/)

[Massimiliano Patacchiola](http://mpatacchiola.github.io)

## Instructions


### Dependences

[Numpy](http://www.numpy.org/)

```shell
sudo pip install numpy
```

[OpenCV](https://opencv.org/)

```shell
sudo apt-get install libopencv-dev python-opencv
```

[TensorFlow](https://www.tensorflow.org/)

```shell
sudo pip install tensorflow
```

[Google Cloud Vision API](https://cloud.google.com/vision/) - **Please note you need a Google Cloud account to import the python modules and run the code**
[Pgmpy](http://pgmpy.org/)

### Installation

Download the repository from [[here]](https://github.com/lukeoverride/deemotions/archive/master.zip) or clone it using git:

```shell
git clone https://github.com/lukeoverride/deemotions.git
cd deemotions/wild
```

### Test with pre-trained weights

To run the code, please type on a terminal:

`python ClassifyImage.py <test-images-path> <real-label (Positive|Negative|Neutral)>`

Example:

`python ClassifyImage.py /home/yourUsername/images/positiveInstances/ Positive`

`python ClassifyImage.py /home/yourUsername/images/negativeInstances/ Negative`

`python ClassifyImage.py /home/yourUsername/images/neutralInstances/ Neutral`

**Training**

The code used to train the model is available in ./wild/64net_emotion_detection_training.py for demonstration purposes only. 

To launch it, you would need the dataset and additionals .csv files. Unless you are interested in increasing global warming, I do not think it is a good idea.

