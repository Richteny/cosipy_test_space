.. _Documentation:

===============
Getting started
===============

.. _requirements:


Requirements
============

<<<<<<< HEAD
For Python 3.11 and above, the recommended environment manangers are conda/mamba.

Pre-requisites
--------------

Install GDAL:

.. code-block:: bash

    sudo apt-get install gdal-bin libgdal-dev
    pip install --upgrade gdal==`gdal-config --version` pybind11  # with pip
    conda install gdal  # with conda/mamba

If you are installing dependencies with conda/mamba, use ``-c conda-forge`` if it does not already have the highest channel priority.

When you are installing from source, :ref:`the provided makefile<makefile>` will install the ``gdal`` package automatically.

.. note:: If you are installing with **pip** and Python 3.11+, you will need to `compile and install richdem`_.
    This is not necessary when using conda/mamba.

.. _`compile and install richdem`: https://github.com/r-barnes/richdem?tab=readme-ov-file#compilation

The ``icc_rt`` package may provide a performance boost on some systems:

.. code-block:: bash

    pip install icc-rt             # with pip
    conda install icc_rt -c numba  # with conda/mamba

Installation with pip
---------------------

Installing COSIPY as a package allows it to run from any directory:

.. code-block:: bash

    pip install cosipymodel
    conda install richdem  # if you are using conda/mamba and python 3.11+
    cosipy-setup  # generate sample configuration files
    cosipy-help   # view help

This is the recommended installation method if you do not plan to modify the source code.

Installation from Source
------------------------

Activate your preferred python environment, then install dependencies:

.. code-block:: bash

    git clone https://github.com/cryotools/cosipy.git
    cd cosipy

    make install-conda-envs                      # install using conda/mamba
    conda install --file conda_requirements.txt  # install with conda

    pip install -r requirements.txt              # install default environment
    pip install -r dev_requirements.txt          # install dev environment

    python3 COSIPY.py -h
    make commands      # if you prefer less typing
    make setup-cosipy  # generate configuration files

Installation as an Editable
---------------------------

Installing COSIPY as an editable allows it to run from any directory.

The :ref:`provided makefile<makefile>` can simplify your workflow.
View all possible commands using ``make help``.

.. code-block:: bash

    git clone https://github.com/cryotools/cosipy.git
    cd cosipy

    make install            # with conda/mamba
    make install-pip        # with pip

    cosipy-setup            # generate sample configuration files
    cosipy-help             # view help

That's it!
Other installation options with pip:

.. code-block:: bash

    pip install -e .        # identical to make install-pip
    make install-pip-tests  # install with test dependencies using pip
    make install-pip-dev    # install with development dependencies using pip

    cosipy-setup            # generate sample configuration files
    cosipy-help             # view help

.. _upgrading:

Upgrading from an Older Version of COSIPY
-----------------------------------------

COSIPY 2.0 is not backwards-compatible with previous versions of COSIPY.
If you have written your own modules that import from ``constants.py``, ``config.py``, or use Slurm, these will break.

Navigate to COSIPY's root directory and convert your existing configuration files:

.. code-block:: bash
=======
Packages and libraries
----------------------
>>>>>>> 8ea90ec ((Update): Update to many PRs from COSIPY release 2.0. Bulk-changes)

COSIPY should run with any Python 3 version on any operating system. If you think the
reason for a problem might be your specific Python 3 version or your operating
system, please create a topic in the forum. The model is tested and
developed on:

 * Anaconda Distribution on max OS
 * Python 3.6.5 on Ubuntu 18.04
 * Anaconda 3 64-bit (Python 3.6.3) on CentOS Linux 7.4
 * High-Performance Cluster Erlangen-Nuremberg University 

The model requires the following libraries:

 * xarray
 * netcdf4
 * numba
 * dask_jobqueue
 * numpy (included in Anaconda)
 * pandas (included in Anaconda)
 * scipy (included in Anaconda)
 * distributed (included in Anaconda)


Additional packages (optional):

 * gdal (e.g. in Debian-based Linux distributions package called gdal-bin)
 * climate date operators (e.g. in Debian-based Linux distributions package called cdo)
 * netCDF Operators (e.g. in Debian-based Linux distritutions package called nco)


.. _tutorial:

<<<<<<< HEAD
Tutorial
========

For this tutorial, download or copy the sample ``data`` folder and place it in your COSIPY working directory.
If you have installed COSIPY as a package, use the entry point ``setup-cosipy`` to generate the sample configuration files.
Otherwise, run ``make setup-cosipy``.
=======
Quick tutorial
==============
>>>>>>> 8ea90ec ((Update): Update to many PRs from COSIPY release 2.0. Bulk-changes)

Pre-processing
--------------

COSIPY requires a file with the corresponding meteorological and static input
data. Various tools are available to create the file from simple text or
geotiff files.


.. _static_tutorial:

Create the static file
~~~~~~~~~~~~~~~~~~~~~~~

In the first step, topographic parameters are derived from the Digital Terrain
Model (DEM) and written to a NetCDF file. A shape file is also required to
delimit the glaciated areas. The DEM and the shapefile should be in lat/lon
WGS84 (EPSG:4326) projection.

.. note:: The DEM can be reprojected to EPSG:4326 using gdal::

           > gdalwarp -t_srs EPSG:4326 dgm_hintereisferner.tif dgm_hintereisferner-lat_lon.tif 


COSIPY comes with the script create_static_file.py located in the utilities folder.
This script runs some gdal routines in the command line. That's is the reason that
we can provide this script only for UNIX and MAC users at the moment.
The script creates some intermediate NetCDF files (dem.nc, aspect.nc,
mask.nc and slope.nc) that are automatically deleted after the static file is created. 

Here we use the DEM **n30_e090_3arc_v2.tif** (SRTM) and the shapefile
**Zhadang_RGI6.shp** provided in the /data/static folder. The static file is
created using::

        python create_static_file.py

The command creates a new file **Zhadang_static.nc** in the /data/static folder.
The file names and paths can be simply changed in the python script.

<<<<<<< HEAD
    make create-static  # from source
    python -m cosipy.utilities.createStatic.create_static_file  # from source
    cosipy-create-static  # from entry point

The command creates a new file **Zhadang_static.nc** in the ``./data/static/`` folder.
=======
>>>>>>> 8ea90ec ((Update): Update to many PRs from COSIPY release 2.0. Bulk-changes)

.. _input_tutorial:

Create the COSIPY input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The creation of the input file requires the static information (file) from
:ref:`section <static_tutorial>`. To convert the data from an automatic weather station
(AWS) we use the conversion script aws2cosipy.py located in the folder
/utilities/aws2cosipy. The script comes with a configuration file
aws2cosipyConfig.py which defines the structure of the AWS file and other
user-defined parameter. Since the input file provides point information, the
data is interpolated via lapse rates for two-dimensional runs.  The solar
radiation fields is based on a model suggested by Wohlfahrt et al.  (2016; doi:
10.1016/j.agrformet.2016.05.012).  Other variables as wind velocity and cloud
cover fraction are assumed to be constant over the domain.

.. note:: The script aws2cosipy.py only serves to illustrate how data can be
          prepared for COSIPY. For most applications it is recommended to develop your
          own routine for data interpolation.

The script is executed with

::

        > python aws2cosipy.py / 
          -c ../../data/input/Zhadang/Zhadang_ERA5_2009_2018.csv / 
          -o ../../data/input/Zhadang/Zhadang_ERA5_2009.nc /
          -s ../../data/static/Zhadang_static.nc /
          -b 20090101 -e 20091231

+-----------+-------------+
| Argument  | Description |
+-----------+-------------+
| -c        | meteo file  |
+-----------+-------------+
| -o        | output file |
+-----------+-------------+
| -s        | static file |
+-----------+-------------+
| -b        | start date  |
+-----------+-------------+
| -e        | end date    |
+-----------+-------------+

<<<<<<< HEAD
The example should take about a minute on a workstation with 4 cores.
=======
If the script was executed successfully, the file
/data/input/Zhadang/Zhadang_ERA5_2009.nc should have been created.
>>>>>>> 8ea90ec ((Update): Update to many PRs from COSIPY release 2.0. Bulk-changes)

.. _run:

Execute the COSIPY model:
~~~~~~~~~~~~~~~~~~~~~~~~~

To run Cosipy, run the following command in the root directory::

<<<<<<< HEAD
.. _makefile:

Makefile
--------

The provided makefile can simplify your workflow.
View all possible commands using ``make help``.

Available shortcuts:
    :black:                 Format all python files with black.
    :build:                 Build COSIPY package.
    :commands:              Display help for COSIPY.
    :commit:                Test, then commit.
    :coverage:              Run pytest with coverage.
    :docs:                  Build documentation.
    :flake8:                Lint with flake8.
    :format:                Format all python files.
    :help:                  Display this help screen.
    :install-conda-env:     Install conda/mamba dependencies.
    :install:               Install editable package using conda/mamba.
    :install-pip-all:       Install editable package with tests & documentation using pip.
    :install-pip-dev:       Install editable package in development mode using pip.
    :install-pip-docs:      Install editable package with local documentation using pip.
    :install-pip:           Install editable package using pip.
    :install-pip-tests:     Install editable package with tests using pip.
    :isort:                 Optimise python imports.
    :pkg:                   Run tests, build documentation, build package.
    :pylint:                Lint with Pylint.
    :run:                   Alias for ``make commands``.
    :setup-cosipy:          Generate COSIPY configuration files.
    :tests:                 Run tests.

.. _configuration:
=======
        > python COSIPY.py
>>>>>>> 8ea90ec ((Update): Update to many PRs from COSIPY release 2.0. Bulk-changes)

The example should take about 3-5 minutes on a workstation with 4 cores.

.. note:: **The configuration and definitions of parameters/constants is done
          in config.py and constants.py.**


Visualization
--------------

     
