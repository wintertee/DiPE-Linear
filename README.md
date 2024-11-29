# Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2411.17257-B31B1B.svg?logo=arxiv)](https://arxiv.org/abs/2411.17257)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2411.17257-FAB70C.svg?logo=DOI)](https://doi.org/10.48550/arXiv.2411.17257)
[![license](https://img.shields.io/github/license/wintertee/DiPE-Linear?style=flat)](https://github.com/wintertee/DiPE-Linear/blob/main/LICENSE)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-etth1-720-1)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720-1?p=disentangled-interpretable-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-etth2-720-1)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-720-1?p=disentangled-interpretable-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-ettm1-720-1)](https://paperswithcode.com/sota/time-series-forecasting-on-ettm1-720-1?p=disentangled-interpretable-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-ettm2-720-1)](https://paperswithcode.com/sota/time-series-forecasting-on-ettm2-720-1?p=disentangled-interpretable-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-weather-720)](https://paperswithcode.com/sota/time-series-forecasting-on-weather-720?p=disentangled-interpretable-representation-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/disentangled-interpretable-representation-for/time-series-forecasting-on-electricity-720)](https://paperswithcode.com/sota/time-series-forecasting-on-electricity-720?p=disentangled-interpretable-representation-for)

The official implementation of paper "Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting"

## Requirements

We recommend using the latest versions of dependencies. However, you can refer to the `environment.yml` file to set up the same environment as we used.

## Dataset

All datasets are stored as CSV files and compressed in GZ format. Please place the datasets in the `./dataset` directory.

- For the M5 dataset, we recommend downloading it from [M5-methods](https://github.com/Mcompetitions/M5-methods) and preprocessing it using `preprocessing/M5.py`. 
- For other datasets, we recommend downloading them from [Autoformer](https://github.com/thuml/Autoformer).

## Usage

All experiments can be reproduced using the `scripts/DiPE.sh` script.

## Citation

If you find this repo useful, please cite our paper:

```bibtex
@misc{zhao2024dipe,
      title={Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting}, 
      author={Yuang Zhao and Tianyu Li and Jiadong Chen and Shenrong Ye and Fuxin Jiang and Tieying Zhang and Xiaofeng Gao},
      year={2024},
      eprint={2411.17257},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.17257}, 
}
```

## License

This repo is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ETDataset](https://github.com/zhouhaoyi/ETDataset)
- [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)
- [Autoformer](https://github.com/thuml/Autoformer)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- [RTSF](https://github.com/plumprc/RTSF)
- [FITS](https://github.com/VEWOXIC/FITS)
