# Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat #

[Shantanu Ghosh](https://shantanu48114860.github.io/),
[Ke Yu](https://gatechke.github.io/),
[Forough Arabshahi](https://forougha.github.io/),
[Kayhan Batmanghelich](https://www.batman-lab.com/)
<br/>
BU ECE, Pitt ISP, META AI, BU ECE <br/>
In [ICML, 2023](https://icml.cc/Conferences/2023/Dates) <br/>

### Table of Contents

1. [Objective](#objective)
2. [Environment setup](#causal-tracing)
3. [Rank-One Model Editing (ROME)](#rank-one-model-editing-rome-1)
4. [CounterFact](#counterfact)
5. [Evaluation](#evaluation)
    * [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)
    * [Integrating New Editing Methods](#integrating-new-editing-methods)
6. [How to Cite](#how-to-cite)

### Objective

In this paper, we aim to blur the dichotomy of explaining a Blackbox post-hoc and building inherently interpretable by
design models. Beginning with a Blackbox, we iteratively \emph{carve out} a mixture of interpretable experts (MoIE) and
a \emph{residual network}. Each interpretable model specializes in a subset of samples and explains them using First
Order Logic (FOL). We route the remaining samples through a flexible residual. We repeat the method on the residual
network until all the interpretable models explain the desired proportion of data.
