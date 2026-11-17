Contributing to Cornac
======================

This page walks you through setting up Cornac for development and opening a pull request.

Prerequisites
^^^^^^^^^^^^^

Before contributing to Cornac, make sure you have Git and a GitHub account.

Fork the Repository
^^^^^^^^^^^^^^^^^^^
Click the "Fork" button on the Cornac GitHub repository
to create a copy of the repository in your GitHub account.

Clone Your Fork
^^^^^^^^^^^^^^^
Clone your forked repository to your local machine using the following command:

.. code-block:: bash

    git clone https://github.com/your-github-id/cornac.git
    cd cornac

Setting Up Your Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a virtual environment, build Cornac and run the tests with the commands
in `Development Setup <guidelines.html#development-setup>`__.

Now you are ready to contribute to Cornac!

Create a New Branch
^^^^^^^^^^^^^^^^^^^

Create a new branch for your contribution:

.. code-block:: bash

    git checkout -b my-feature-branch


Make Your Changes
^^^^^^^^^^^^^^^^^

Make your code changes, bug fixes, or improvements to Cornac.


Testing Your Changes
^^^^^^^^^^^^^^^^^^^^

Build Cornac on your environment, then run the tests and an example to check that your changes
are working as expected.

1. If you changed a Cython (``.pyx``) file, build Cornac on your environment again:

   .. code-block:: bash

       python setup.py build_ext --inplace

2. Run the tests:

   .. code-block:: bash

       pytest

   See `Tests <guidelines.html#tests>`__ for which tests to add with your change
   and how to run the dataset tests.

3. Run an example that uses the code you changed.


Commit Your Changes
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


Creating a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^

Once your feature is ready, open a pull request for review and merging.

1. Visit your fork on GitHub.
2. Click the "New Pull Request" button.
3. Choose the base repository (PreferredAI/cornac) and branch (master) for your pull request.
4. Write a clear and detailed pull request description.
5. Click "Create Pull Request."

GitHub's guide to `creating a pull request
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`__
has more details.

:doc:`/developer/guidelines` sums up what we check when we review a pull request.

Your pull request will be reviewed by the Cornac maintainers.
Please be patient during the review process, and be prepared to address any feedback.


Communication
^^^^^^^^^^^^^

Issues
~~~~~~

If you encounter bugs or have ideas for improvements, create an issue on
the GitHub issue tracker at https://github.com/PreferredAI/cornac/issues.

Review Process
~~~~~~~~~~~~~~

Your pull request will be reviewed by Cornac maintainers.
They will provide feedback and request changes if necessary.

As this is an open source project, the repository is maintained on a voluntary
basis. We thank you for your patience during the review process.

License
~~~~~~~

By contributing to Cornac, you agree that your code will be released under the Apache 2.0 License.
Make sure to include the appropriate license headers in your files.

**Thank you for contributing to Cornac!
Your contributions are greatly appreciated and help make Cornac a
better tool for everyone.**
