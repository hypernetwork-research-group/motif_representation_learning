/opt/miniconda/bin/conda init
/opt/miniconda/bin/conda create -n .conda python=3.10 -y

/opt/miniconda/bin/conda install -n .conda pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
/opt/miniconda/bin/conda install -n .conda pyg -c pyg -y
/opt/miniconda/bin/conda install -n .conda pytorch-scatter -c pyg -y

/opt/miniconda/bin/conda run -n .conda python3 -m pip install -r requirements.txt
/opt/miniconda/bin/conda run -n .conda python3 -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
/opt/miniconda/bin/conda run -n .conda python3 setup.py build_ext --inplace
