# Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat #

### [Project Page]() | [Paper]() | [Arxiv](https://arxiv.org/pdf/2302.10289.pdf)

[Shantanu Ghosh <sup>1</sup>](https://shantanu48114860.github.io/),
[Ke Yu <sup>2</sup>](https://gatechke.github.io/),
[Forough Arabshahi <sup>3</sup>](https://forougha.github.io/),
[Kayhan Batmanghelich <sup>1</sup>](https://www.batman-lab.com/)
<br/>
<sup>1</sup> BU ECE, <sup>2</sup> Pitt ISP, <sup>3</sup> META AI <br/>
In [ICML, 2023](https://icml.cc/Conferences/2023/Dates) <br/>

### Table of Contents

1. [Objective](#objective)
2. [Environment setup](#environment-setup)
3. [Downloading data](#rank-one-model-editing-rome-1)
4. [Data preprocessing](#counterfact)
5. [Training pipleline](#evaluation)
    * [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)
    * [Integrating New Editing Methods](#integrating-new-editing-methods)
6. [How to Cite](#how-to-cite)

### Objective

In this paper, we aim to blur the dichotomy of explaining a Blackbox post-hoc and building inherently interpretable by
design models. Beginning with a Blackbox, we iteratively *carve out* a mixture of interpretable experts (MoIE) and a *
residual network*. Each interpretable model specializes in a subset of samples and explains them using First Order
Logic (FOL). We route the remaining samples through a flexible residual. We repeat the method on the residual network
until all the interpretable models explain the desired proportion of data.

<img src='images/method.gif'><br/>

### Environment setup