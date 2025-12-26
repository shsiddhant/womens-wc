# Women's Cricket World Cup 2025 Outcome Prediction

## Overview

The project aims to predict match outcomes for the recently concluded **Women's Cricket World Cup
2025** using historical match data for matches between Women's Cricket World Cup 2022 and Women's Cricket World Cup 2025.

Raw ball-by-ball data is sourced from [Cricsheet](https://cricsheet.org/) and
transformed into match level feature dataset, each row representing a match.

---
## Problem Statement

Given match-level features derived from ball-by-ball data, can we predict the outcome of a Women’s Cricket World Cup match?

This is formulated as a **supervised learning** problem, where historical matches
are used to train a model that will be evaluated on matches from the Women’s
Cricket World Cup 2025.

---
## Installation

### 1. Clone the repository

```shell
git clone https://github.com/shsiddhant/womens-wc.git
```

###  2. Install Dependencies

#### A. Using uv

```shell
uv venv .venv --seed
source .venv/bin/activate
uv sync
```

#### B. Using pip

```shell
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Using the notebooks

Go to the cloned directory. Then activate the environment and run JupyterLab.

```shell
source .venv/bin/activate
jupyter lab notebooks/
```

---
## Data Source and Handling

### Raw Data

- **Source:** [Cricsheet](https://cricsheet.org/)
- **Format:** JSON (one file per match)
- **Type:** Ball-by-ball
- **Count:** 159 (some will be ignored, see below for explanation)
- **Scope:** 
	- Only the matches between the eight teams that participated in the Women's Cricket World Cup 2025 will be considered.
	- Only matches after World Cup 2022 are considered.

To keep the repository lightweight, raw data files are not included in the repository. You can obtain it with the following steps.

### Obtaining the Raw Data

1. Download zipped JSON data from [Cricsheet](https://cricsheet.org/downloads/) for WODIs.

2. Unzip and place the JSON files inside `data/raw`.

---
## Data Processing

### Base Dataset

The base dataset is a **match-level dataset**, where each row represents a single 
match. Raw ball-by-ball JSON data is parsed and transformed to create the base dataset.

The base dataset contains match information such as team names, venue, date, etc. and
foundational match statistics such as runs scored, wickets taken, deliveries played.
From these, a match-level feature dataset is derived.

Both the base dataset and feature dataset have been saved as parquet files, and can be found inside `data/processed` - `base_dataset.parquet` and
`features_dataset.parquet`.
You can generate them from raw data using the data processing scripts (as explained in the notebook) .

#### Base Dataset Schema

**Note:** The indexing 0 and 1 is decided alphabetically. So if the match is between, say India and England, then team_0 would be England and team_1 would be India.

| Column Name      | Description                                                                                                               |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------ |
| match_id         | A unique id for the match derived from file name. For example, a match id 1490443 corresponds to the file '1490443.json'. |
| country          | The country where the match venue is located.                                                                             |
| start_date       | Match start date.                                                                                                         |
| team_0           | The first team playing the match (teams ordered alphabetically).                                                          |
| team_1           | The second team playing the match.                                                                                        |
| toss_winner      | Toss winner - 0 if team_0 wins the toss, otherwise 1.                                                                     |
| toss_decision    | Decision taken by the team that won the toss - 0 if they decide to bat first, otherwise 1.                                |
| runs_0 (1)       | Runs scored by team_0 (1)                                                                                                 |
| wickets_0 (1)    | Wickets lost by team_0 (1)                                                                                                |
| deliveries_0 (1) | Deliveries played by team_0 (1)                                                                                           |
| result           | Result of the game - 0 if team_0 wins and 1 if team_1 wins. Tied matches and matches with no result are not considered.   |

---

## Features (So far)

1. Home Advantage

2. Weighted Batting Averages and Strike-rates

3. Weighted Bowling Averages and Economy.

4. Weighted Win Percentage.

The last three are calculated for both teams.

**Note:** Feature set may evolve in future with additions to model venue conditions,
spin/pace bowling strengths etc.

---
## Roadmap

- [x] Build Features.
- [ ] Exploratory Data Analysis
- [ ] Train simple ML models
- [ ] Evaluate model performance
- [ ] Document results

---
## Tools and Libraries

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- jupyter


---
## License
[![LICENSE: MIT](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)
