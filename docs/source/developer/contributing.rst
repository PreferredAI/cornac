Contributing to Cornac
======================

Fork the Repository
^^^^^^^^^^^^^^^^^^^
Click the "Fork" button on the Cornac GitHub repository
to create a copy of the repository in your GitHub account.

Clone Your Fork
^^^^^^^^^^^^^^^
Clone your forked repository to your local machine using the following command:

.. code-block:: bash

    git clone https://github.com/your-github-id/cornac.git

Create a New Branch
^^^^^^^^^^^^^^^^^^^

Create a new branch for your contribution:

.. code-block:: bash

    git checkout -b my-feature-branch


Make your Changes
^^^^^^^^^^^^^^^^^

Make your code changes, bug fixes, or improvements to Cornac.


Testing your Changes
^^^^^^^^^^^^^^^^^^^^

Build Cornac on your environment, followed by testing it on an example to ensure your changes
are working as expected.

1. To build Cornac on your environment:

.. code-block:: bash

    python3 setup.py clean
    python3 setup.py install


.. note::

    The following packages are required for building Cornac on your environment: ``Cython``, ``numpy``, ``scipy``.
    
    If you do not have them, install by using the following commands:

    .. code-block:: bash

        pip3 install Cython numpy scipy

2. Run an example depending on what you are fixing/enhancing on.


Commit your changes
^^^^^^^^^^^^^^^^^^^

Commit your changes with a descriptive commit message:

.. code-block:: bash

    git add .
    git commit -m "Add new feature: <Your Feature Name>"


Push to Your Fork
^^^^^^^^^^^^^^^^^

Push your changes to your GitHub fork:

.. code-block:: bash

    git push origin my-feature-branch


Creating a pull request
^^^^^^^^^^^^^^^^^^^^^^^

Once your feature is ready for reviewing, do a pull request for review and merging.

1. Visit your fork on GitHub.
2. Click the "New Pull Request" button.
3. Choose the base repository (PreferredAI/cornac) and branch (main) for your pull request.
4. Write a clear and detailed pull request description.
5. Click "Create Pull Request."

View this guide for more information on how to do a pull request.
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request


Your pull request will be reviewed by the Cornac maintainers.
Please be patient during the review process, and be prepared to address any feedback.



Development Guidelines
======================

Documentation
^^^^^^^^^^^^^

Contributions should include relevant and concise documentation.
This includes docstrings, comments, and updates to the official documentation when needed.


Communication
=============

Issues
^^^^^^

If you encounter bugs or have ideas for improvements, create an issue on
the GitHub issue tracker at https://github.com/PreferredAI/cornac/issues.

Review Process
^^^^^^^^^^^^^^

Your pull request will be reviewed by Cornac maintainers.
They will provide feedback and request changes if necessary.

As this is an open source project, the repository is maintained on a voluntary
basis. We thank you for your patience during the review process.

License
^^^^^^^

By contributing to Cornac, you agree that your code will be released under the Apache 2.0 License.
Make sure to include the appropriate license headers in your files.

**Thank you for contributing to Cornac!
Your contributions are greatly appreciated and help make Cornac a
better tool for everyone.**

