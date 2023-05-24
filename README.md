# Dequantile

This repository contains the code needed to reproduce the experiments from the paper:


> Decorrelation with conditional normalizing flows.

## Dependencies

The base container for this project was built using the recipe in the docker file.

## Data

The data can be downloaded with

```angular2html
wget -P dequantile/downloads/ https://zenodo.org/record/3606767/files/W_high_level.npz
```

## Boosted W tagging

Run `python experiments/boosted_w.py`.

## Decorrelate existing
This code demonstrates how to run the conditional flow decorrelation on top of the predictions made by an existing pipeline.

Run `python experiments/dequantile_existing.py`.

## Repo set up
```angular2html
docker pull gitlab-registry.cern.ch/saklein/dequantile/latest
```

Set up the python package, this must be run from the top level of the project

```angular2html
docker run --rm -it -v ${PWD}:/workspace gitlab-registry.cern.ch/saklein/dequantile/latest bash -c "chmod +777 run_install.sh; ./run_install.sh"
```

For launching jobs on the cluster, you will need to add

```angular2html
export PYTHONPATH=${PWD}:${PWD}/python_install:${PYTHONPATH}
```

to the start of every job.

To get the singularity image on the cluster you will need to

```angular2html
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12
cd container
singularity pull gitlab-registry.cern.ch/saklein/dequantile/latest
cd ..
singularity shell -B ${PWD} container/latest_latest.sif
chmod +777 run_install.sh
./run_install.sh
```
