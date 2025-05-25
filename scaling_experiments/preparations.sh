#!/bin/bash
BRANCH=${1:-main}

# fail on error, unset variable,
# or if any stage in a pipeline fails
set -euo pipefail

echo Downloading dataset from gdrive...

pip install gdown &> /dev/null

gdown --folder \
  https://drive.google.com/drive/folders/1ZnnuHvAvLLVLiNcwLxkuxMkma0EppUa0 \
  -O ./original_dataset

dataset_path=$(realpath original_dataset)

echo Cloning repositories \& installing dependencies ...

echo Cloning $BRANCH of weaver-core...
git clone --branch "$BRANCH" --depth 1 https://github.com/michael-2956/weaver-core.git &> /dev/null
echo Cloning $BRANCH of particle_transformer...
git clone --branch "$BRANCH" https://github.com/michael-2956/particle_transformer.git &> /dev/null
echo Installing weaver-core...
pip install -e weaver-core &> /dev/null

echo Creating env.sh...

echo "#!/bin/bash" > particle_transformer/env.sh
echo >> particle_transformer/env.sh
echo "export DATADIR_JetClass=" >> particle_transformer/env.sh
echo "export DATADIR_TopLandscape=${dataset_path}" >> particle_transformer/env.sh
echo "export DATADIR_QuarkGluon=" >> particle_transformer/env.sh

chmod +x ./particle_transformer/sync_weaver.sh
chmod +x ./particle_transformer/start_model.sh

echo
echo Ready to run!
echo Now, cd particle_transformer and run each script from scaling_experiments/setting_collections individually
echo For example:
echo cd particle_transformer
echo ./scaling_experiments/setting_collections/setting_1.sh
echo ...
echo ./scaling_experiments/setting_collections/setting_10.sh
