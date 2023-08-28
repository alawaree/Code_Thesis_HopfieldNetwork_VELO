
VELO tracking at the LHCb experiment using Hopfield networks
============================================================

This master research thesis explores the amenability Hopfield networks for track reconstruction of the VELO detector in the LHCb experiment. It investigates an algorithm introduced by a MRP at Maastricht University, while increasing data volume, modifying functions and tuning parameters. The aim is to optimize the algorithm's performance under different data conditions.


This study is based on two previous projects:

 - The first project was created by Dr D. Campora who develops the SOTA algorithm for track reconstruction in the VELO detector of the LHCb experiment at CERN. We will use the *event model* to parse the inut files specified in json and the *validator* to validate our results.

 - The second project is a master research project from Maastricht Universtiy that experimented 3 different types of track reconstruction algorithms. One of their implementation was using Hopfield networks to reconstruct particle tracks in the VELO detector, this implementation is used in our project as a starting point for our work.


Quick summary about the content of each file:
---------------------------------------------

- data_analysis: data analysis of the datasets, algorithms used to generate events and graphs
- datasets: minibias and bsphiphi datasets containing events used to perform the track reconstruction  
- event_model: functions defining what is an event, a track, a hit, a module
- experiments: a lot of algorithms added as well as the results of some experiments
- results: text files regrouping the results of the different experiments
- results_analysis: analysis of the various results obtained
- validator: validator built to validate the results
- visual : differents visualisers of the results

