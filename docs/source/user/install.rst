Installation
===================

This page shows you how to install Cornac with pip, conda or from source, and
how to add the extra packages that some models need.

Install Python
--------------
Cornac supports Python 3.10 or newer. If you have not done so, go to the official `Python download page <https://www.python.org/downloads/>`_.

Install Cornac
--------------
There are 3 different ways in which you could install cornac.
Depending on your environment and requirements, choose and run the
corresponding code to install Cornac:

1. Using ``pip`` (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pip is a package manager for Python. It allows users to easily install,
update, and manage third-party libraries and frameworks that are available
on the Python Package Index (PyPI).

.. code-block:: bash

    pip3 install cornac

2. Using ``conda``
^^^^^^^^^^^^^^^^^^

Conda is an open-source package management system and environment
management system for installing, creating, and managing software
environments on Windows, macOS, Linux, and other operating systems.

.. code-block:: bash

    conda install cornac -c conda-forge

3. From GitHub source - For latest updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Should you require the latest updates of Cornac from GitHub,
you could manually build and install using the following code, which needs a
C/C++ compiler:

.. code-block:: bash

    pip3 install git+https://github.com/PreferredAI/cornac.git

Model dependencies
------------------

Certain models in Cornac may require additional dependencies.
The requirements.txt file shows what dependencies are required for each model.

Take the model VAECF for example.

.. code-block::
    :caption: models/vaecf/requirements.txt https://github.com/PreferredAI/cornac/blob/master/cornac/models/vaecf/requirements.txt

    torch>=0.4.1


In order to utilize this model, this dependency needs to be installed.
To install all dependencies in a model's requirements.txt file, give pip the
file's address on GitHub. You do not need a copy of the source code:

.. code-block:: bash

    pip3 install -r https://raw.githubusercontent.com/PreferredAI/cornac/master/cornac/models/vaecf/requirements.txt


.. admonition:: Note for macOS users

    Some algorithm implementations use OpenMP to support multi-threading.
    The macOS wheels on PyPI are built without OpenMP, so if you install Cornac
    with ``pip``, those algorithms run on a single thread. For multi-threading,
    install ``libomp`` from Homebrew, then install Cornac
    `from GitHub source <3. From GitHub source - For latest updates_>`__, which
    turns OpenMP on when it finds ``libomp``:

    .. code-block:: bash

        brew install libomp

Verifying Installation
----------------------
After installing Cornac, you can verify that it has been successfully installed
by running the following command on your favourite terminal/command prompt:

.. code-block:: bash

    python3 -c "import cornac; print(cornac.__version__)"

You should see the following output:

.. parsed-literal::
    |release|

Congratulations! Your machine has Cornac and you're now ready to
create your first experiment!



What's Next?
------------
Start creating your first experiment by following the :doc:`/user/quickstart` guide.