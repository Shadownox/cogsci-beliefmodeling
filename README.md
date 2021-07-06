CogSci Belief Modeling
======================

Companion repository for the 2021 article "Unifying Models for Belief and Syllogistic Reasoning" published in the proceedings of the 43rd Annual Meeting of the Cognitive Science Society.

### Overview

- `benchmark`: Contains the CCOBRA benchmark file.
- `benchmark/evaluation.json`: Benchmark file used to perform a coverage CCOBRA-analysis.
- `data`: Contains the CCOBRA data version of the Trippas-2018 dataset and the extraction script.
- `data/raw_Trippas2018`: Contains a readme file with a link to the original repository. Place the data from OSF here.
- `data/extract.py`: Extracts the information from the dataset located in `data/raw_Trippas2018` and converts it to a CCOBRA dataset.
- `data/Trippas2018.csv`: CCOBRA version of the Trippas-2018 dataset.
- `models`: Contains the models.
- `models/caches`: Contains caches for the possible and necessary responses used by the mreasoner model.
- `models/helpers`: Contains a helper class for PHM.
- `models/fol.py`: Implementation of a first-order-logic-based model for syllogistic reasoning.
- `models/MisinterpretedNecessity.py`: Implementation of the misinterpreted necessity model for the belief effect.
- `models/mreasoner.py`: Model providing the responses of [mReasoner](https://www.modeltheory.org/models/mreasoner) based on a cached results for different parameter configurations.
- `models/NoBeliefModel.py`: Meta-model ignoring the belief effect.
- `models/phm.py`: Implementation of PHM.
- `models/portfolio.py`: Model selecting the best belief model and reasoning model for each individual participant.
- `models/portfolio_belief.py`: Model selecting the best belief model for a fixed reasoning model for each individual participant.
- `models/Random.py`: A model responding with a random response.
- `models/SelectiveScrutinyModel.py`: Implementation of the selective scrutiny model for the belief effect.
- `models/UserMedian.py`: Model responding with the median rating of the respective participant.
- `plots`: Contains scripts and data for replicating the plots.
- `plots/results`: Contains the datasets obtained from the benchmark used to generate the plots.
- `plots/results/2021-01-26_models_only.csv`: Data containing only the cognitive models (excludes the portfolio approaches).
- `plots/results/2021-01-27-with_portfolio.csv`: Data containing all models.
- `plots/plot_improvement.py`: Plots Figure 2 from the paper.
- `plots/plot_perf.py`: Plots Figure 3 and Figure 4 from the paper.

### Dependencies

- Python 3
    - [CCOBRA](https://github.com/CognitiveComputationLab/ccobra)
    - [pandas](https://pandas.pydata.org)
    - [numpy](https://numpy.org)
    - [seaborn](https://seaborn.pydata.org)

### Run the data extraction script

First, download the [Trippas-2018 dataset](https://osf.io/kt3jn/) and place it into the folder `data/raw_Trippas2018`. You can then generate the CCOBRA-data by running the following command:

```
cd /path/to/repository/data/
$> python extract.py
```

The file will be placed in the same folder as the script, named `Trippas2018.csv`.


### Run the benchmark

After installing CCOBRA, run the following command to execute the benchmark:

```
cd /path/to/repository/analysis/benchmark/
$> ccobra evaluation.json
```

An HTML-file will be created in the same folder. When opening the file, the predictive performance of the models is shown with a possibility to save the results as a csv-file.

### Run the plotting scripts

After running the benchmark and saving the results as a csv-file, the plotting scripts can be executed with the following commands:

```
cd /path/to/repository/plots/
$> python plot_improvement.py [path/to/results.csv]
$> python plot_perf.py.py [path/to/results.csv]
```

If no file is provided, the default result-files from the article are used. The plots will be placed as pdf-files in the same directory as the scripts.

### References

Brand, D., Riesterer, N., and Ragni, M. (in press). Unifying models for belief and syllogistic reasoning. In Proceedings of the 43th Annual Meeting of the Cognitive Science Society, eds. T. Fitch, C. Lamm, H. Leder, and K. T. mar Raible

Trippas, D., Kellen, D., Singmann, H., Pennycook, G., Koehler, D. J., Fugelsang, J. A., & Dub´e, C. (2018). Characterizing belief bias in syllogistic reasoning: A hierarchical Bayesian meta-analysis of ROC data. Psychonomic Bulletin & Review, 25(6), 2141–2174.


