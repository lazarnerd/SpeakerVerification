# /bin/bash

sudo chown -R vscode:vscode /home/vscode

poetry config virtualenvs.create true
poetry config virtualenvs.in-project true