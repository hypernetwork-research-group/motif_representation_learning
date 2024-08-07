# Hypergraph Motifs Representation Learning

This repository contains datasets and codes for the paper "Hypergraph Motifs Representation Learning".

## How to Install

Install all the requirements specified in the *requirements.txt* file.

`pip install -r requirements.txt`

Then install Pytorch and PyTorch and PyTorch Geometric libraries.

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install torch_geometric
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

Build all the Cython code for the motif counting and the negative sampling.

`python3 setup.py build_ext --inplace`

## How to Run the Experiments



In order to run the experiment of h-motif prediction on the $k=2$ motif of the *email-Enron* dataset and the ranking based negative sampling, run the following command:

`python3 . -k 2 --dataset email_Enron --mode rank`

In the same way as for the original MoCHy implementation, the number associated to the h-motif (k) does not directly correspond to the h-motif index. Following table describes the number that should be used to refer to a specific h-motif index.

| k   | 2 | 3 | 5 | 7 | 8 | 9 | . | . | . |
|-----|---|---|---|---|---|---|---|---|---|
| h-motif index | 1 | 2 | 3 | 4 | 5 | 6 | . | . | . |

