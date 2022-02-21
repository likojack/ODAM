# ODAM

Implementation for ODAM: Object Detection, Association, and Mapping using Posed RGB Video

## Requirements

1. install the packages using the following command:

`conda env create -f environment.yml`

2. Copy [ScanNet](http://www.scan-net.org/) data to ./data

3. Download the pretrained model from [here](https://drive.google.com/drive/folders/13tpl9j0TGuJjXBCmsyLqHWBque27n-xv?usp=sharing) and place them in ./experiments/

## Run ODAM

run the full pipeline using the following command at the root dir:

`export PYTHONPATH=$PYTHONPATH:$PWD`

`python src/scripts/run_processor.py --config_path ./configs/detr_scan_net.yaml --no_code --use_prior --out_dir ./result/e2e --representation super_quadric`


Note: The comparison to Vid2CAD reported in the paper did not report the best performance of Vid2CAD due to some inconsistencies in the representation. See the [updated Vid2CAD](https://arxiv.org/pdf/2012.04641.pdf) for the latest results.
