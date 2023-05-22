# Spatial community structure impedes language amalgamation in a population-based iterated learning model
This repository contains the code used to generate data for [Spatial community structure impedes language amalgamation in a population-based iterated learning model](https://google.com) (Accepted as a presentation at [Artificial Life 2023](https://2023.alife.org/)).
## Setup
Create the directory ```data/```
## Population Model
To run the population model:
```
python popModel.py -u utterences
```
with the following arguments
| Parameter      | Description |
| ----------- | ----------- |
| -u utterences     | The number of utterances spoken per generation. Default is 50.  |

Output will be two files in ```data/```, the stability and expressiveness for each agent for each generation.
## Community Model
To run the community model:
```
python comModel.py -u utterences
```
with the following arguments
| Parameter      | Description |
| ----------- | ----------- |
| -p percent     | The percent quantity of language being external. Default is 1.  |
| -s     | Spatial. Adds spatial structure to communities. Default is Off.  |
| -v     | Verbose. Adds more print tracking of agents communication, stability, expressiveness and timer. Default is Off.  |

Output will be two files in ```data/```. The average expressiveness and stability per generation as well as the final state of the model stored in a ```data/stabilityP.csv``` where P is the percent of external communication. This file is a 900 length csv which stores the stability between all agents, so the first 30 are the stability between first agent and all agents, second 30 are the stability between second agent and all agents e.t.c.
