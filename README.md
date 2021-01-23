# Classification-and-localization-of-fish-species
This project aims to perform a deep learning based classification and regression, in order to classify fish species and also to calculate the bounding box, in which the fishes locate. This project is gotten from a Kaggle competition, accessible at: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/overview

This project asks to detect which species of fish appears on a fishing boat, based on images captured from boat cameras of various angles.
The goal of the project is:
1) Detecting the species of the fish in a fishing boat, based on images captured from various angles. - Classes: `ALB`, `BET`, `DOL`, `LAG`, `SHARK`, `YFT`, `OTHER`, `NoF` (no fish). Hence, we have a classification problem, with 8 target classes.
2) Localization of the fishes on the boat, in which we should solve a regression problem to calculate 4 numbers (coordinates of the surrounding bounding box).

Dataset:
You can find and download the dataset in this link:
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data
Moreover, the annotations for the dataset is provided in the annotation folder, which is downloaded from competition link (provided by one of the users in JSON file).


In order to make the code run, set the right path to the datset in the main.py code.

This project has been done as class project, inspired by other feely available online codes. Feel free to change any of the hyper parameters to get the best accuracy.
