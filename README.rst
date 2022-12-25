PurPliance
=========================================

This repository contains the component that extracts privacy-statement tuples
from policy sentences of PurPliance.


Run code
--------

Download and extract the NER model
####################################

Download NER model `en_core_web_lg.high_f1_data_org.model.tar.xz <https://drive.google.com/file/d/1DVFJKIjJ1VdcKgaT9aAuYQonTwvzr76c/view?usp=sharing>`_
and extract the file to ``src/oppnlp/analyze/pded/models/``:

.. code-block:: bash

  src/oppnlp/analyze/pded/models $ tar -xvf en_core_web_lg.high_f1_data_org.model.tar.xz


Set up dependencies and run test code
#####################################

To run the code, use a virtual environment and run the test as follows:

.. code-block:: bash

  # Create new conda python environment.
  conda create -n purpliance_oss python=3.8
  conda activate purpliance_oss

  # Install the current package.
  pip install -e .
  pip install -r requirements.txt

  # Test: Extract privacy statements from each file in test/policies and output
  # as json files in test/policies/stmt.
  bash test/test.sh

  # Extract privacy statements from each file in the $INPUT_DIR
  # json files in $INPUT_DIR/stmt.
  # INPUT_DIR contains plain sentencized text files *.txt to be analyzed.
  # Each non-blank line in the file should contain one and only 1 sentence.
  python src/runner/analyze/priv_stmt/run_priv_stmt_extractor.py $INPUT_DIR

The code was tested on Ubuntu 18.04 and MacOS 12.6.

Publication
-----------

| Duc Bui, Yuan Yao, Kang G. Shin, Jong-Min Choi, and Junbum Shin.
| Consistency Analysis of Data-Usage Purposes in Mobile Apps.
| In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS '21).
| Association for Computing Machinery, New York, NY, USA, 2824–2843.
| https://doi.org/10.1145/3460120.3484536


License
-------

PurPliance is licensed under the BSD-3-Clause License (See `LICENSE.txt <LICENSE.txt>`_).


Acknowledgement
---------------

This repo uses code from `PolicyLint/PoliCheck repository <https://github.com/benandow/PrivacyPolicyAnalysis>`_.
