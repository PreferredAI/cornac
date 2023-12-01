Getting Started
===============

Prerequisites
^^^^^^^^^^^^^

Before contributing to Cornac, make sure you have the following prerequisites:
- Python 3.x
- Git
- Virtual Environment (optional but recommended)


Setting Up Your Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fork the Cornac repository on GitHub.
2. Clone your fork to your local machine.
3. Create a virtual environment and install the necessary packages using pip.

.. code-block:: bash

    git clone https://github.com/your-github-id/cornac.git
    cd cornac
    python -m venv venv
    source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
    pip install -r requirements.txt

Now you are ready to contribute to Cornac!

Contributing
^^^^^^^^^^^^

Cornac is an open-source project and we welcome contributions of all kinds.
We welcome pull requests for new features such as models, datasets, metrics
and anything that could make Cornac a better tool for all.


How to Contribute
^^^^^^^^^^^^^^^^^

View :doc:`/developer/contributing` for more information on how to contribute.


Creating a new model
^^^^^^^^^^^^^^^^^^^^

View https://github.com/PreferredAI/cornac/blob/master/tutorials/add_model.md
for more information on how to create a new model.

Adding a new metric
^^^^^^^^^^^^^^^^^^^

View https://github.com/PreferredAI/cornac/blob/master/tutorials/add_metric.md
for more information on how to add a new metric.
