Installation
===================

Install Python
--------------
Cornac supports most versions of Python 3. If you have not done so, go to the official `Python download page <https://www.python.org/downloads/>`_.

Install Cornac
--------------
There are 3 different ways in which you could install cornac.
Depending on your environment and requirements, choose and run the
corresponding codes to install Cornac:

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
you could manually build and install using the following codes:

.. code-block:: bash

    pip3 install git+https://github.com/PreferredAI/cornac.git

Model dependencies
------------------

Certain models in Cornac may require additional dependencies.
The requirements.txt file shows what dependencies are required for each model.\

Take the model WMF for example.

.. code-block::
    :caption: models/wmf/requirements.txt https://github.com/PreferredAI/cornac/blob/master/cornac/models/wmf/requirements.txt

    tensorflow==2.12.0


In order to utilize this model, this dependency needs to be installed.
To install all dependencies in a provided requirements.txt file, follow these steps:

1. Using your favourite terminal/command prompt, navigate to the models in which you want to utilize

.. code-block:: bash

    cd cornac/models/wmf

2. Install the dependencies by using this command:

.. code-block:: bash

    pip3 install -r requirements.txt


.. admonition:: Note for MacOS users

    Some algorithm implementations use OpenMP to support multi-threading.
    For MacOS users, in order to run those algorithms efficiently, you might need to install gcc from Homebrew to have an OpenMP compiler:

    .. code-block:: bash

        brew install gcc | brew link gcc

Verifying Installation
----------------------
After installing Cornac, you can verify that it has been successfully installed
by running the following command on your favourite terminal/command prompt:

.. code-block:: bash

    python3 -c "import cornac; print(cornac.__version__)"

You should see the following output:

.. parsed-literal::
    |version|

Congratulations! Your machine has Cornac and you're now ready to
create your first experiment!



What's next?
------------
Start creating your first experiment by following the :doc:`quickstart` guide.