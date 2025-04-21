echo "Preparing necessary files from https://github.com/NVIDIA/kvpress.git"
mkdir tmp-kvpress && cd tmp-kvpress
git clone https://github.com/NVIDIA/kvpress.git
mv ./kvpress/kvpress ../
mv ./kvpress/evaluation ../ 
cd ..
rm -rf ./tmp-kvpress


