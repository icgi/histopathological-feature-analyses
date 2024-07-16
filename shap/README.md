# Shapely analysis

This repository contains a Docker container for running SHAP (SHapley Additive exPlanations) analysis using the ExplainableBoostingRegressor model. The container uses Python 3 and includes all necessary libraries for the analysis.

## SHAP Analysis

SHAP (SHapley Additive exPlanations) is a method to explain individual predictions. It connects optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory and their related extensions. This container utilizes the ExplainableBoostingRegressor from the `interpret` library to create interpretable models and generate SHAP plots.

The analysis involves:
1. **Loading the data**: Independent variables and the dependent variable (percentage of tiles for one of 100 clusters).
2. **Training the model**: Using ExplainableBoostingRegressor to fit the data.
3. **Generating SHAP plots**: Creating beeswarm and waterfall plots to visualize feature importance and contributions.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Building

Build the docker container with `docker-compose`.

```bash
docker-compose build shap
```

To run the analysis:
```bash
docker-compose run shap python3 src/shap_analysis.py
```

### Acknowledgements
- SHAp library: https://github.com/shap/shap
- InterpretML: https://github.com/interpretml/interpret
