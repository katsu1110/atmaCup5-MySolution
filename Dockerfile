FROM jupyter/tensorflow-notebook:5811dcb711ba

LABEL maintainer="katsu1110"

USER root

# --- Install python-tk htop python-boost
RUN apt-get update && \
    apt-get install -y --no-install-recommends python-tk software-properties-common htop libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Install dependency gcc/g++
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

# --- Install gcc/g++
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc-7 g++-7 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Update alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# --- Conda xgboost, lightgbm, catboost
RUN conda install --quiet --yes \
    'boost' \
    'lightgbm' \
    'xgboost' \
    'catboost' \
     && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN conda install -c conda-forge seaborn

# --- Install vowpalwabbit, optuna, sklearn-deap, yellowbrick, spacy
RUN $CONDA_DIR/bin/python -m pip install vowpalwabbit \
					 optuna \
					 update_checker \
					 tqdm \
                     --no-warn-conflicts -q tensorflow-addons==0.9.1 \

###########
#
# Add some usefull libs, inspired by kaggle's Dockerfile
# CREDITS : https://hub.docker.com/r/kaggle/python/dockerfile
#
###########
#RUN $CONDA_DIR/bin/python -m pip install --upgrade mpld3
RUN $CONDA_DIR/bin/python -m pip install mplleaflet \
                     statsmodels \
                     astropy \
					 gpxpy \
					 arrow \
					 Geohash \
					 haversine \
					 toolz cytoolz \
					 sacred \
					 plotly \
					 fitter \
					 langid \
# Delorean. Useful for dealing with datetime
					 delorean \
					 trueskill \
					 heamy \
					 vida \
# Useful data exploration libraries (for missing data and generating reports)
					 missingno \
					 pandas-profiling \
					 s2sphere
###########
#
# Issue #1
# pandas.read_hdf
#
###########
RUN $CONDA_DIR/bin/python -m pip install --upgrade tables

# clean up pip cache
RUN rm -rf /root/.cache/pip/*