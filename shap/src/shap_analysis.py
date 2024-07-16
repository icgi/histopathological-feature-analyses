# Import necessary libraries
import pandas as pd
import shap
from interpret.glassbox import ExplainableBoostingRegressor
import matplotlib.pyplot as plt

def load_data(file_path, res):
    """
    Load the data from file path for the resolution.
    
    Args:
    file_path (str): The path to the input file containing the independent and dependent variables.
    res (str): Resolution [highres/lowres].
    
    Returns:
    X (pd.DataFrame): Feature variables.
    y (pd.Series): Target score.
    """
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Assuming the dependent variable is the last column
    print(df.keys())
    features = [x for x in df.keys() if x.startswith("percent")]
    X = df[features]
    y = df[f"{res}_score"]
    
    return X, y

def train_model(X, y):
    """
    Train an ExplainableBoostingRegressor model.
    
    Args:
    X (pd.DataFrame): The independent variables.
    y (pd.Series): The dependent variable.
    
    Returns:
    model (ExplainableBoostingRegressor): The trained model.
    """
    # Initialize the model
    model = ExplainableBoostingRegressor(interactions=0)

    # Fit the model
    model.fit(X, y)
    
    return model

def create_shap_plots(model, X, res):
    """
    Generate SHAP beeswarm and waterfall plots for the model.
    
    Args:
    model (ExplainableBoostingRegressor): The trained model.
    X (pd.DataFrame): The feature variables.
    res (str): Resolution [highres/lowres].
    """
   
    # Sample 200 instances for use as the background distribution
    X200 = shap.utils.sample(X, 200)
    
    # Create a SHAP explainer
    explainer = shap.Explainer(model.predict, X200)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Generate beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=100)
    plt.title("SHAP Beeswarm Plot")
    plt.savefig(f"output/beeswarm_{res}.pdf", dpi=300, bbox_inches="tight")
    plt.clf()
    
    # Generate waterfall plot for the first example
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=100)
    plt.title("SHAP Waterfall Plot for Example 1")
    plt.savefig(f"output/waterfall_{res}.pdf", dpi=300, bbox_inches="tight")
    plt.clf()

def main(file_path, res):
    """
    Main function to load data, train model, and create SHAP plots.
    
    Args:
    file_path (str): The path to the input data
    """
    # Load the data
    X, y = load_data(file_path, res)
    
    # Train the model
    model = train_model(X, y)
    
    # Create SHAP plots
    create_shap_plots(model, X, res)

# Run SHAP analysis
if __name__ == "__main__":
    input_file_paths = dict(
            lowres="data/Galaxy_uni_results_uni_lowres_max-per-patient-100_kmeans-100.xlsx",
            highres="data/Galaxy_uni_results_uni_highres_max-per-patient-1000_kmeans-100.xlsx"
            )

    for res, input_file_path in input_file_paths.items():
        main(input_file_path, res)

