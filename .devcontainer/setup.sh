# /bin/bash

echo "alias ll='ls -lah'" >> ~/.bashrc

sudo chown -R vscode:vscode /home/vscode

sudo apt update -y
sudo apt upgrade -y

pip install -r requirements.txt