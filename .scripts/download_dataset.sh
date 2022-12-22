#!/usr/bin/env bash
#==================================================================
ROOT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)")"
#==================================================================

# Download the dataset
source ${ROOT_DIR}/.env

DATA_PATH=${ROOT_DIR}/data


mkdir ${DATA_PATH}
python ${ROOT_DIR}/dataprep.py --save_path ${DATA_PATH} --download --user ${USERNAME} --password ${PASSWORD} 