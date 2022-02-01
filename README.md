# Analysing Reciprocity in Code Review

This is Team Balloon's project for the [MSR Virtual Hackathon 2022.](https://conf.researchr.org/track/msr-2022/msrhackathon2022#msr-virtual-hackathon-2022)

We analysed reciprocity in the code review process. Using [vector autoregressive (VAR) models](https://en.wikipedia.org/wiki/Vector_autoregression) 
over GitHub's pull request data, we explored if there is a causal relationship between: 1) reviews performed by a mantainer 
and 2) the acceptance of their own code contributions.

This repository contains the Python scripts for performing such analysis.

## Installation

The code was tested on MacOS using Anaconda Python 3.9 and ElasticSearch 7.15.2. You need to 
[install ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) in your 
system before running the scripts.
Before running the scripts, please [start ElasticSearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html) and update the global variable `ELASTICSEARCH_HOST` on the
`config.py` file with your ElasticSearch URL.

To install Python dependencies, execute the following:

```
pip install -r requirements.txt
```

Finally, we use GitHub's API ,via [Perceval](https://github.com/chaoss/grimoirelab-perceval), to obtain 
pull-request data.
For this to work, we need you to obtain a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
from your GitHub account and set it to the global variable `GITHUB_API_TOKEN` on the `config.py` file.

## Usage

The file `dataloading.py` contains functions for retriving pull-request from GitHub and
storing it in ElasticSearch. The following snippet retrieves 
pull-requests for the [Kubernetes repository](https://github.com/kubernetes/kubernetes), created over the last 10 years:

```python
from dataloading import get_and_store

def star_extraction(owner: str, repository: str):
    get_and_store(owner, repository, factor=0, new_index=True)
    for year_factor in range(1, 10):
        get_and_store(owner, repository, factor=year_factor, new_index=False)


if __name__ == "__main__":
    star_extraction(owner="kubernetes", repository="kubernetes")
```
This code will store the data in a ElasticSearch index called `kubernetes-kubernetes`.
To perform the reciprocity analysis over this project, we can use the functions from
the `devanalysis.py` file:

```python
from typing import Tuple

import elasticsearch
from elasticsearch import Elasticsearch

from aggregation import PRS_REVIEWED_AND_MERGED, PRS_AUTHORED_AND_MERGED
from config import ELASTICSEARCH_HOST
from devanalysis import analyse_project

if __name__ == "__main__":
    elastic_search: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    variables: Tuple = (PRS_REVIEWED_AND_MERGED, PRS_AUTHORED_AND_MERGED)
    project_prs, project_analysis = analyse_project(elastic_search, "kubernetes-kubernetes", "month", variables,
                                                    "aic")
```
This code will generate monthly time series from the data stored in the `kubernetes-kubernetes` index.
Regarding the order of the VAR model, it will choose the value with the minimum Akaike Information Criteria (AIC).
Per developer, plots will be stored at the `img` folder and the output of statistical tests at `txt`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss 
what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)