# Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting

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
