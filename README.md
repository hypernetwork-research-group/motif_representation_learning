# Hypergraph Motifs Representation Learning

This repository contains datasets and codes for the paper **Hypergraph Motifs Representation Learning**.
Alessia Antelmi, Gennaro Cordasco, Daniele De Vinco, Valerio Di Pasquale, Mirko Polato, Carmine Spagnuolo,
KDD 2025

## Overview

Hypergraphs have emerged as a powerful tool for representing
high-order connections in real-world complex systems. Similar to
graphs, local structural patterns in hypergraphs, known as high-
order motifs (h-motifs), play a crucial role in network dynamics and
serve as fundamental building blocks across various domains. For
this reason, predicting h-motifs can be highly beneficial in different
fields. In this paper, we aim to advance our understanding of such
complex high-order dynamics by introducing and formalizing the
problem of h-motifs prediction. To address this task, we propose a
novel solution that leverages both high-order and pairwise informa-
tion by combining hypergraph and graph convolutions to capture
hyperedges correlation within h-motifs. A key component of our
solution is the generation of non-trivial negative samples, designed
to generate close-to-positive negative samples. To evaluate the ef-
fectiveness of our approach, we defined several baselines inspired
by existing literature on hyperedge prediction methods. Our ex-
tensive experimental assessments demonstrate that our approach
consistently outperforms all the considered baselines, showcasing
its superior performance and robustness in predicting h-motifs.

## How to Run HGMRL

We used the Python version 3.10.14 to run the experiments.

### Manually Install

Install all the requirements specified in the *requirements.txt* file.

`pip install -r requirements.txt`

Then install Pytorch and PyTorch and PyTorch Geometric libraries.

**CUDA 12.1**

```bash
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
conda install pytorch-scatter -c pyg -y
conda run -n .conda python3 -m pip install -r requirements.txt
conda run -n .conda python3 -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
conda run -n .conda python3 setup.py build_ext --inplace
```

Build all the Cython code for the motif counting and the negative sampling.

`python3 setup.py build_ext --inplace`

### Using Docker

If you prefer, you can use Docker to set up and run the project.

```bash
# on arm
docker build --build-arg PLAT=linux/aarch64 -t hgmrl .
# on x86
docker build --build-arg PLAT=linux/amd64 -t hgmrl .

docker run -it hgmrl
```

Then inside the container run the following command to activate the conda environment: `conda activate .conda`.

### Execute

```bash
usage: . [-h] -k K [--dataset {email_enron,contact_high_school,contact_primary_school,cora}] [--limit LIMIT] [--mode {rank,random,prob}]

options:
  -h, --help            show this help message and exit
  -k K
  --dataset {email_enron,contact_high_school,contact_primary_school,cora}
  --limit LIMIT
  --mode {rank,random,prob}
```

## How to reproduce the experiments

In order to run the experiment of h-motif prediction on the $k=2$ motif of *email-Enron*, *cora*, *contact-High-School* and *contact-Primary-School* datasets with the ranking based negative sampling, run the following command:

```bash
python3 . -k 2 --dataset email_enron --mode rank
python3 . -k 2 --dataset cora --mode rank
python3 . -k 2 --dataset contact_high_school --mode rank
python3 . -k 2 --dataset contact_primary_school --mode rank
```

In the same way as for the original [MoCHy](https://github.com/geon0325/MoCHy) implementation, the number associated to the h-motif (k) does not directly correspond to the h-motif index. Following table describes the number that should be used to refer to a specific h-motif index.

| k   | 2 | 3 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h-motif index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |

## Data format

contact-High-School, contact-Primary-School and email-Enron datasets are from the *Simplicial closure and higher-order link prediction.
Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg.
Proceedings of the National Academy of Sciences (PNAS), 2018* repository.

Hypergraph data are stored inside the `datasets/__datsets__` folder, each consisting of 3 main files:

- {DATASET}-nverts.txt | number of vertices contained in the hypergraph at that row;
- {DATSAET}-simplices.txt | list of vertices;
- {DATASET}-times.txt | timestamp associated to the hyperedge.

All three files represent a vector of integers. There is one integer per line.

The first file contains the number of vertices within each simplex. The second
file is a contiguous list of the nodes comprising the simplices, where the
ordering of the simplices is the same as in the first file. The third file
contains the timestamps for each simplex. The length of the vectors in the first
and third files is the same, and the length of the vector in the second file is
the sum of the integers in the first file.

Consider an example dataset consisting of three simplices:
1. {1, 2, 3} at time 10
2. {2, 4} at time 15.
3. {1, 3, 4, 5} at time 21.
Then files would look as follows:

example-nverts.txt
3
2
4

example-simplices.xt
1
2
3
2
4
1
3
4
5

example-times.txt
10
15
21

## License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/hypernetwork-research-group/motif_representation_learning">Hypergraph Motifs Representation Learning</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/hypernetwork-research-group">Alessia Antelmi, Gennaro Cordasco, Daniele De Vinco, Valerio Di Pasquale, Mirko Polato, Carmine Spagnuolo</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p>