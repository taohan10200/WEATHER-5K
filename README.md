#  WEATHER-5K Benchmark 
WEATHER-5K dataset consists of a comprehensive collection of data from 5,672 weather stations worldwide, spanning a 10-year period with one-hour intervals. It includes multiple crucial weather elements (temperature, dewpint temperature, wind speed, wind rate, sea level pressure), providing a more reliable and interpretable resource for forecasting.

<p align="center">
<img src=".\asset\Overview.png" height = "400" alt="" align=center />
</p>


:triangular_flag_on_post:**News** (2024.06)  We release the WEATHER-5K as a comprehensive benchmark, allowing for a thorough evaluation of time-series forecasting methods and facilitates advancements in this field.

## Leaderboard of WEATHER-5K benchmark

Until now, we have bnchmarked the following models in this repo:
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **Corrformer** - nterpretable weather forecasting for worldwide stations with a unified deep model [[NMI 2023]](https://www.nature.com/articles/s42256-023-00667-9) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Corrformer.py).

  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).


| Model<br>Ranking | Long-term<br>Forecasting                          | Short-term<br>Forecasting                                    | Imputation                                                   | Classification                                         |  Anomaly<br>Detection                                    |
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [iTransformer](https://arxiv.org/abs/2310.06625)  | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            |   [PatchTST](https://github.com/yuqinie98/PatchTST)    | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [TimesNet](https://arxiv.org/abs/2210.02186) | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |

<!-- 
**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible. -->

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[OneDrive]](https://hkustconnect-my.sharepoint.com/:f:/g/personal/thanad_connect_ust_hk/EhYDpJPhvixNkfJXXZkKOxcBxTa7ckEE70c65x8PKsYRKQ?e=kSvrrf), Then place and `unzip` the downloaded data in the folder`./dataset`. 


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# Global Station Weather Forecasting
bash ./scripts/weather-5k/iTransformer.sh

```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

<!-- ## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
``` -->

## Contact
If you have any questions or suggestions, feel free to contact:

- Tao Han (hantao10200@gmail.com)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the [Time-Series-Library](https://github.com/thuml/Time-Series-Librar). We sincerely thank the contributors for their contributions.


