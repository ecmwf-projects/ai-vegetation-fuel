Training and Inference using Docker
-----------------------------------

This guide assumes you have ``Docker`` and ``docker-compose`` installed
and setup to run as non-root user following the instructions
`here <https://docs.docker.com/engine/install/>`__,
`here <https://docs.docker.com/engine/install/linux-postinstall/>`__ and
`here <https://docs.docker.com/compose/install/>`__.

Steps
~~~~~

-  Clone the repository.
-  Download the data and place it in a ``data/`` directory at the root
   of the repository.
-  Navigate to the ``docker/`` directory.
-  Run ``export UID=$(id -u)`` and then ``export GID=$(id -g)``.
-  Run ``docker-compose up --build`` which will build the image, run a
   container and launch a Jupyter server on port ``4242``.
-  Use the link in the Jupyter command output to access any of the
   several notebooks for EDA, Training, Inference and Error Analysis.
-  If you would like to run the CLI interface, use
   ``docker-compose run ml-fuel bash`` to launch an interactive
   terminal.
-  You can now run ``pre-processing.py``, ``train.py`` or ``test.py``
   located in the ``src/`` directory. Check the docs for more details.

The steps above mount the local code repository and data directory to a
volume on the container, setting up the correct permissions so that you
can keep any pretrained models or inference files even after the
container is shut down.
