# fluidflower_benchmark_image_analysis
DarSIA image analysis of FluidFlower International Benchmark dataset.


A. Preliminaries:

The FluidFlower International Benchmark dataset can be found at:
Kris Eikehaug et al., The International FluidFlower benchmark study
dataset (2023), https://zenodo.org/record/7510589#.Y9pKo3bMKQ4

Once donwloaded, it can be analyzed using the code included in this
repository, allowing to reproduce the results in:
Martin A. Fernø et al., Room-scale CO2 injections in a physical
reservoir model with faults (2023), https://arxiv.org/abs/2301.06397

In order to run the analysis, one has to download and install
DarSIA:
Jakub W. Both et al., DarSIA v1.0 (2023),

For an introduction to DarSIA, we refer to:
Jan M. Nordbotten et al., DarSIA: An open-source Python toolbox
for two-scale image processing of dynamics in porous media (2023),
https://arxiv.org/abs/2301.05455


B. Instructions:

  1. In the main directory of the repo, run: 'python setup.py develop'.

  2. Update the file 'image_analysis/data.json'. Follow the instructions
  listed in the template.

  3. Run './run.sh', producing intermediate results as cache, and storing
  all main results in the results folder specified in 'data.json'.
  These include in particular those required for the sparse data analysis.
  In addition some spatia;l maps are created on coarser meshed with 1cm 
  grid size.

  
