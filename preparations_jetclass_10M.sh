#!/bin/bash
BRANCH=${1:-main}

# fail on error, unset variable,
# or if any stage in a pipeline fails
set -euo pipefail

echo Cloning repositories \& installing dependencies ...

echo Cloning $BRANCH of weaver-core...
git clone --branch "$BRANCH" --depth 1 https://github.com/michael-2956/weaver-core.git &> /dev/null
echo Cloning $BRANCH of particle_transformer...
git clone --branch "$BRANCH" https://github.com/michael-2956/particle_transformer.git &> /dev/null
echo Installing weaver-core...
pip install -e weaver-core &> /dev/null

echo Downloading dataset \(10M\)...

cd particle_transformer && ./get_datasets.py JetClass-10M && cd ..

if [[ -e ./particle_transformer/sync_weaver.sh ]]; then
  chmod +x ./particle_transformer/sync_weaver.sh
fi
if [[ -e ./particle_transformer/start_model.sh ]]; then
  chmod +x ./particle_transformer/start_model.sh
fi

echo
echo Ready to run!
