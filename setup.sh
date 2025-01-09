PLATFORM=$1

echo "Installing on $PLATFORM" >> log
if [ $PLATFORM == "linux/aarch64" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /opt/miniconda.sh
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh
fi
    
bash /opt/miniconda.sh -b -p /opt/miniconda
/opt/miniconda/bin/conda init
/opt/miniconda/bin/conda create -n .conda python=3.10 -y

if [ $PLATFORM == "linux/aarch64" ]; then
    apt install g++ -y
    /opt/miniconda/bin/conda install -n .conda pytorch==2.3.0 cpuonly -c pytorch -y
    /opt/miniconda/bin/conda run -n .conda pip install pybind11
    /opt/miniconda/bin/conda run -n .conda pip install "pybind11[global]"
    /opt/miniconda/bin/conda run -n .conda python3 -m pip install torch_geometric
    /opt/miniconda/bin/conda run -n .conda python3 -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
    /opt/miniconda/bin/conda run -n .conda python3 -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
else
    /opt/miniconda/bin/conda install -n .conda pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    /opt/miniconda/bin/conda install -n .conda pyg -c pyg -y
    /opt/miniconda/bin/conda install -n .conda pytorch-scatter -c pyg -y
    /opt/miniconda/bin/conda run -n .conda python3 -m pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
fi

/opt/miniconda/bin/conda run -n .conda python3 -m pip install -r requirements.txt
/opt/miniconda/bin/conda run -n .conda python3 setup.py build_ext --inplace

if [ $PLATFORM == "linux/aarch64" ]; then
    /opt/miniconda/bin/conda install -n .conda -c conda-forge libstdcxx-ng -y
fi 
