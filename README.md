#  WEATHER-5K Benchmark 
## Paper: [WEATHER-5K: A Large-scale Global Station Weather Dataset Towards Comprehensive Time-series Forecasting Benchmark](https://arxiv.org/abs/2406.14399)

## Introduction

The WEATHER-5K dataset is a large-scale time series forecasting dataset containing weather data from 5,672 weather stations worldwide. It is a valuable resource for researchers and developers in the field of time-series forecasting, providing a comprehensive evaluation of various methods and models.
WEATHER-5K dataset consists of a comprehensive collection of data from 5,672 weather stations worldwide, spanning a 10-year period with one-hour intervals. It includes multiple crucial weather elements (temperature, dewpint temperature, wind speed, wind rate, sea level pressure), providing a more reliable and interpretable resource for forecasting.

<p align="center">
<img src=".\asset\Overview.png" height = "350" alt="" align=center />
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


<!-- 
**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible. -->

## Benchmarking results 

The results are reported at 4 different prediction lengths: 24, 72, 120, and 168, where the input length is 48. 
<table>
  <colgroup>
    <col>
    <col>
    <col style="border-left: 1px solid black;">
    <col style="border-right: 1px solid black;">
    <col style="border-left: 1px solid black;">
    <col style="border-right: 1px solid black;">
    <col style="border-left: 1px solid black;">
    <col style="border-right: 1px solid black;">
    <col style="border-left: 1px solid black;">
    <col style="border-right: 1px solid black;">
    <col style="border-left: 1px solid black;">
    <col style="border-right: 1px solid black;">
  </colgroup>
    <tr>
        <td>Baselines<strong></td>
        <td>Lead Time</td>
        <th colspan="2">Temperature</th>
        <th colspan="2">Dewpoint</th>
        <th colspan="2">Wind Speed</th>
        <th colspan="2">Wind Direction</th>
        <th colspan="2">Sea Level Pressure</th>
        <th colspan="2">Overall</th>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>MAE</td>
        <td>MSE</td>
        <td>MAE</td>
        <td>MSE</td>
        <td>MAE</td>
        <td>MSE</td>
        <td>MAE</td>
        <td>MSE</td>
        <td>MAE</td>
        <td>MSE</td>
        <td>MAE</td>
        <td>MSE</td>
    </tr>
<tr>
        <td rowspan="4">🥇 1st Pyraformer</td>
        <td>24</td>
        <td><strong>1.75<strong></td>
        <td><strong>6.92<strong></td>
        <td><strong>1.83<strong></td>
        <td><strong>7.88<strong></td>
        <td><strong>1.30<strong></td>
        <td><strong>3.58<strong></td>
        <td>61.8</td>
        <td>6930.2</td>
        <td><strong>1.90<strong></td>
        <td><strong>9.72<strong></td>
        <td>13.7</td>
        <td>1391.7</td>
    </tr>
    <tr>
        <td>72</td>
        <td><strong>2.47<strong></td>
        <td><strong>13.03<strong></td>
        <td><strong>2.67<strong></td>
        <td><strong>15.39<strong></td>
        <td><strong>1.52<strong></td>
        <td>4.97</td>
        <td>72.0</td>
        <td><strong>8222.4<strong></td>
        <td><strong>3.76<strong></td>
        <td><strong>33.67<strong></td>
        <td>16.5</td>
        <td>1657.9</td>
    </tr>
    <tr>
        <td>120</td>
        <td><strong>2.77<strong></td>
        <td><strong>16.04<strong></td>
        <td><strong>3.00<strong></td>
        <td><strong>18.95<strong></td>
        <td>1.59</td>
        <td><strong>5.37<strong></td>
        <td><strong>75.1<strong></td>
        <td>8610.7</td>
        <td><strong>4.43<strong></td>
        <td><strong>43.91<strong></td>
        <td><strong>17.4<strong></td>
        <td>1739.0</td>
    </tr>
    <tr>
        <td>168</td>
        <td><strong>2.95<strong></td>
        <td><strong>17.95<strong></td>
        <td><strong>3.20<strong></td>
        <td><strong>21.06<strong></td>
        <td>1.61</td>
        <td>5.56</td>
        <td>76.4</td>
        <td>8773.5</td>
        <td><strong>4.77<strong></td>
        <td><strong>49.97<strong></td>
        <td><strong>17.8<strong></td>
        <td>1773.6</td>
    </tr>
        <tr>
        <td rowspan="4">🥈 2nd iTransformer</td>
        <td>24</td>
        <td>1.82</td>
        <td>7.49</td>
        <td>1.93</td>
        <td>8.80</td>
        <td>1.32</td>
        <td>3.77</td>
        <td>63.2</td>
        <td>7358.8</td>
        <td>1.99</td>
        <td>10.84</td>
        <td>14.1</td>
        <td>1478.0</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.60</td>
        <td>14.46</td>
        <td>2.84</td>
        <td>17.5</td>
        <td><strong>1.52<strong></td>
        <td>4.96</td>
        <td>73.2</td>
        <td>8713.3</td>
        <td>4.14</td>
        <td>40.65</td>
        <td>16.9</td>
        <td>1758.2</td>
    </tr>
    <tr>
        <td>120</td>
        <td>2.97</td>
        <td>18.36</td>
        <td>3.24</td>
        <td>22.16</td>
        <td>1.59</td>
        <td>5.42</td>
        <td>76.4</td>
        <td>9192.2</td>
        <td>4.95</td>
        <td>54.67</td>
        <td>17.8</td>
        <td>1858.6</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.18</td>
        <td>20.64</td>
        <td>3.48</td>
        <td>24.89</td>
        <td>1.64</td>
        <td>5.67</td>
        <td>78.0</td>
        <td>9441.1</td>
        <td>5.36</td>
        <td>62.31</td>
        <td>18.3</td>
        <td>1910.9</td>
    </tr>
    <tr>
        <td rowspan="4">🥉 3rd Informer</td>
        <td >24</td>
        <td>1.88</td>
        <td>7.51</td>
        <td>1.94</td>
        <td>8.30</td>
        <td><strong>1.30<strong></td>
        <td>3.62</td>
        <td><strong>60.7<strong></td>
        <td><strong>6906.9<strong></td>
        <td>2.01</td>
        <td>10.56</td>
        <td><strong>13.6<strong></td>
        <td><strong>1387.4<strong></td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.75</td>
        <td>14.84</td>
        <td>2.86</td>
        <td>17.24</td>
        <td>1.53</td>
        <td><strong>4.86<strong></td>
        <td><strong>71.5<strong></td>
        <td>8251.4</td>
        <td>4.24</td>
        <td>39.24</td>
        <td><strong>16.4<strong></td>
        <td><strong>1631.4<strong></td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.11</td>
        <td>18.21</td>
        <td>3.25</td>
        <td>21.50</td>
        <td>1.60</td>
        <td>5.38</td>
        <td>75.7</td>
        <td><strong>8504.5<strong></td>
        <td>5.15</td>
        <td>54.31</td>
        <td>18.3</td>
        <td><strong>1720.4<strong></td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.24</td>
        <td>20.24</td>
        <td>3.43</td>
        <td>24.89</td>
        <td>1.63</td>
        <td>5.65</td>
        <td><strong>76.2<strong></td>
        <td><strong>8718.4<strong></td>
        <td>5.26</td>
        <td>58.42</td>
        <td>18.1</td>
        <td><strong>1764.4<strong></td>
    </tr>
    <tr>
        <td rowspan="4">Autoformer</td>
        <td>24</td>
        <td>1.93</td>
        <td>8.64</td>
        <td>2.06</td>
        <td>9.57</td>
        <td>1.42</td>
        <td>3.97</td>
        <td>66.5</td>
        <td>7710.0</td>
        <td>2.26</td>
        <td>12.78</td>
        <td>15.2</td>
        <td>1553.4</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.72</td>
        <td>15.14</td>
        <td>2.97</td>
        <td>18.38</td>
        <td>1.54</td>
        <td>5.14</td>
        <td>75.4</td>
        <td>9111.5</td>
        <td>4.25</td>
        <td>42.34</td>
        <td>17.8</td>
        <td>1846.7</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.21</td>
        <td>20.27</td>
        <td>3.34</td>
        <td>23.12</td>
        <td>1.58</td>
        <td>5.73</td>
        <td>79.2</td>
        <td>9143.5</td>
        <td>4.83</td>
        <td>48.88</td>
        <td>18.1</td>
        <td>1868.3</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.43</td>
        <td>21.71</td>
        <td>3.56</td>
        <td>22.55</td>
        <td>1.64</td>
        <td>5.95</td>
        <td>79.8</td>
        <td>9435.8</td>
        <td>5.32</td>
        <td>61.85</td>
        <td>18.5</td>
        <td>1885.7</td>
    </tr>
    <tr>
        <td rowspan="4">FEDformer</td>
        <td>24</td>
        <td>1.98</td>
        <td>8.45</td>
        <td>2.02</td>
        <td>9.25</td>
        <td>1.36</td>
        <td>3.91</td>
        <td>66.0</td>
        <td>7384.1</td>
        <td>2.13</td>
        <td>11.43</td>
        <td>14.7</td>
        <td>1483.4</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.87</td>
        <td>16.50</td>
        <td>3.01</td>
        <td>18.70</td>
        <td>1.59</td>
        <td>5.31</td>
        <td>76.2</td>
        <td>8824.8</td>
        <td>4.15</td>
        <td>37.60</td>
        <td>17.6</td>
        <td>1780.6</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.19</td>
        <td>20.29</td>
        <td>3.36</td>
        <td>23.10</td>
        <td>1.66</td>
        <td>5.71</td>
        <td>79.0</td>
        <td>9143.3</td>
        <td>4.81</td>
        <td>48.86</td>
        <td>18.4</td>
        <td>1848.3</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.35</td>
        <td>22.12</td>
        <td>3.54</td>
        <td>25.21</td>
        <td>1.68</td>
        <td>5.88</td>
        <td>79.7</td>
        <td>9189.2</td>
        <td>5.01</td>
        <td>53.39</td>
        <td>18.7</td>
        <td>1859.2</td>
    </tr>
    <tr>
        <td rowspan="4">Dlinear</td>
        <td>24</td>
        <td>2.71</td>
        <td>13.82</td>
        <td>2.47</td>
        <td>12.36</td>
        <td>1.44</td>
        <td>4.34</td>
        <td>66.6</td>
        <td>8234.5</td>
        <td>3.09</td>
        <td>21.34</td>
        <td>15.3</td>
        <td>1657.3</td>
    </tr>
    <tr>
        <td>72</td>
        <td>3.55</td>
        <td>23.05</td>
        <td>3.48</td>
        <td>22.85</td>
        <td>1.62</td>
        <td>5.37</td>
        <td>75.0</td>
        <td>9250.8</td>
        <td>4.64</td>
        <td>45.83</td>
        <td>17.7</td>
        <td>1869.6</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.90</td>
        <td>27.60</td>
        <td>3.89</td>
        <td>27.72</td>
        <td>1.67</td>
        <td>5.70</td>
        <td>77.3</td>
        <td>9510.6</td>
        <td>5.19</td>
        <td>56.22</td>
        <td>18.4</td>
        <td>1925.6</td>
    </tr>
    <tr>
        <td>168</td>
        <td>4.11</td>
        <td>30.38</td>
        <td>4.11</td>
        <td>30.58</td>
        <td>1.69</td>
        <td>5.88</td>
        <td>78.4</td>
        <td>9630.0</td>
        <td>5.48</td>
        <td>61.73</td>
        <td>18.8</td>
        <td>1951.7</td>
    </tr>
    <tr>
        <td rowspan="4">PatchTST</td>
        <td>24</td>
        <td>2.05</td>
        <td>9.26</td>
        <td>2.16</td>
        <td>10.58</td>
        <td>1.40</td>
        <td>4.20</td>
        <td>66.2</td>
        <td>7765.8</td>
        <td>2.19</td>
        <td>12.54</td>
        <td>14.8</td>
        <td>1560.5</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.82</td>
        <td>16.60</td>
        <td>3.06</td>
        <td>19.96</td>
        <td>1.60</td>
        <td>5.39</td>
        <td>75.2</td>
        <td>9067.8</td>
        <td>4.28</td>
        <td>42.46</td>
        <td>17.4</td>
        <td>1830.5</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.15</td>
        <td>20.32</td>
        <td>3.43</td>
        <td>24.39</td>
        <td>1.66</td>
        <td>5.79</td>
        <td>77.8</td>
        <td>9452.6</td>
        <td>5.09</td>
        <td>57.29</td>
        <td>18.2</td>
        <td>1912.1</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.33</td>
        <td>22.54</td>
        <td>3.63</td>
        <td>26.94</td>
        <td>1.69</td>
        <td>6.00</td>
        <td>79.0</td>
        <td>9638.1</td>
        <td>5.51</td>
        <td>65.3</td>
        <td>18.6</td>
        <td>1951.7</td>
    </tr>
    <tr>
        <td rowspan="4">Corrformer</td>
        <td>24</td>
        <td>1.99</td>
        <td>8.21</td>
        <td>2.09</td>
        <td>9.47</td>
        <td>1.38</td>
        <td>3.83</td>
        <td>66.7</td>
        <td>7832.3</td>
        <td>2.19</td>
        <td>12.39</td>
        <td>14.9</td>
        <td>1584.4</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.74</td>
        <td>15.16</td>
        <td>2.99</td>
        <td>18.40</td>
        <td>1.56</td>
        <td>4.91</td>
        <td>75.6</td>
        <td>9111.7</td>
        <td>4.27</td>
        <td>42.36</td>
        <td>17.8</td>
        <td>1846.7</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.06</td>
        <td>18.63</td>
        <td>3.34</td>
        <td>22.48</td>
        <td>1.61</td>
        <td>5.56</td>
        <td>78.0</td>
        <td>9477.4</td>
        <td>5.08</td>
        <td>57.13</td>
        <td>18.1</td>
        <td>1915.8</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.09</td>
        <td>18.69</td>
        <td>3.36</td>
        <td>22.53</td>
        <td>1.63</td>
        <td>5.69</td>
        <td>78.9</td>
        <td>9636.0</td>
        <td>5.34</td>
        <td>61.83</td>
        <td>18.4</td>
        <td>1938.8</td>
    </tr>
    <tr>
        <td rowspan="4">Mamba</td>
        <td>24</td>
        <td>1.98</td>
        <td>8.59</td>
        <td>2.01</td>
        <td>9.52</td>
        <td>1.37</td>
        <td>4.02</td>
        <td>66.0</td>
        <td>7709.5</td>
        <td>2.21</td>
        <td>12.73</td>
        <td>14.7</td>
        <td>1548.9</td>
    </tr>
    <tr>
        <td>72</td>
        <td>2.79</td>
        <td>16.00</td>
        <td>2.90</td>
        <td>18.11</td>
        <td>1.55</td>
        <td>5.11</td>
        <td>75.1</td>
        <td>8863.9</td>
        <td>4.29</td>
        <td>41.88</td>
        <td>17.3</td>
        <td>1789.0</td>
    </tr>
    <tr>
        <td>120</td>
        <td>3.03</td>
        <td>18.47</td>
        <td>3.18</td>
        <td>21.02</td>
        <td><strong>1.58<strong></td>
        <td>5.28</td>
        <td>76.7</td>
        <td>8931.2</td>
        <td>4.93</td>
        <td>52.56</td>
        <td>17.9</td>
        <td>1805.7</td>
    </tr>
    <tr>
        <td>168</td>
        <td>3.16</td>
        <td>19.88</td>
        <td>3.32</td>
        <td>22.53</td>
        <td><strong>1.59<strong></td>
        <td><strong>5.35<strong></td>
        <td>77.4</td>
        <td>8958.8</td>
        <td>5.21</td>
        <td>57.37</td>
        <td>18.1</td>
        <td>1812.8</td>
    </tr>
</table>
**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[OneDrive]](https://hkustconnect-my.sharepoint.com/:u:/g/personal/thanad_connect_ust_hk/EZGm7DP0qstElZwafr_U2YoBk5Ryt9rv7P31OqnUBZUPAA?e=5r0wEo), Then place and `unzip` the downloaded data in the folder`./WEATHER-5K`. 


3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# Global Station Weather Forecasting
bash ./scripts/weather-5k/iTransformer.sh

```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Citation

If you find WEATHER-5K is useful, please cite our paper.

```
@misc{han2024weather5k,
    title={WEATHER-5K: A Large-scale Global Station Weather Dataset Towards Comprehensive Time-series Forecasting Benchmark},
    author={Tao Han and Song Guo and Zhenghao Chen and Wanghan Xu and Lei Bai},
    year={2024},
    eprint={2406.14399},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Tao Han (hantao10200@gmail.com)

Or describe it in Issues.

## Acknowledgement

This library is constructed based on the [Time-Series-Library](https://github.com/thuml/Time-Series-Librar). We sincerely thank the contributors for their contributions.


