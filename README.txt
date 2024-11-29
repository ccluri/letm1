This is the software/code for the letm1 knockout model.

The mitochondrial model is based on Nazareth model, with some
modifications to include the effects of a spike. This part of the code
is nazaret_mito.py, translated to python my Chaitanya.
-----------------------------------

The code is a standalone software and the source code for the
simulations is provided in this repository.

The end results of the simulated results are included in 'Letm1
values.ods'

-----------------------------------
SOFTWARE REQUIREMENTS
------------------------------------

1) Python 3.10 or higher with compatible numpy, scipy and matplotlib
libraries.

2) No special hardware is necessary.
It was tested on Ubuntu 22.04 machine, with anaconda installation

------------------------------------
INSTALLATION
------------------------------------
No installation required. The python scripts are stand-alone

-------------------------------------
DEMO
------------------------------------


-------------------------------------
INSTRUCTIONS FOR USE
------------------------------------

i) mitochondria.py, has the mitochondrial model of the TCA and ETC
that is slightly modified from the Nazareth model to include the
per-spike costs.

ii) utils.py, has the additional helper functions that are used to
record values during simulations

iii) figure_properties.py, has properties for default values of label
size etc used to generate some to the figures.


-------------------------------------
LICENSE
-------------------------------------
GPL 3.0 License



