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
