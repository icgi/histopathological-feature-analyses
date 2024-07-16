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
    model = ExplainableBoostingRegressor()

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
    # Feature order

    feature_order = dict(
        lowres=[
            7,
            38,
            79,
            73,
            33,
            85,
            46,
            10,
            41,
            37,
            81,
            99,
            68,
            8,
            53,
            88,
            59,
            70,
            82,
            45,
            13,
            89,
            26,
            2,
            77,
            3,
            5,
            90,
            44,
            50,
            31,
            55,
            43,
            17,
            42,
            57,
            28,
            95,
            97,
            84,
            61,
            19,
            22,
            64,
            78,
            34,
            94,
            18,
            91,
            71,
            39,
            21,
            48,
            93,
            63,
            15,
            6,
            11,
            62,
            40,
            9,
            16,
            30,
            66,
            24,
            35,
            0,
            92,
            27,
            49,
            25,
            14,
            52,
            60,
            96,
            23,
            58,
            86,
            20,
            65,
            76,
            54,
            12,
            74,
            29,
            98,
            80,
            67,
            69,
            72,
            36,
            87,
            32,
            83,
            4,
            56,
            75,
            47,
            51,
            1,
        ],
        highres=[
            65,
            99,
            32,
            72,
            9,
            93,
            89,
            18,
            57,
            43,
            27,
            7,
            96,
            28,
            34,
            84,
            23,
            24,
            33,
            64,
            69,
            45,
            51,
            94,
            59,
            79,
            40,
            42,
            95,
            8,
            91,
            87,
            0,
            56,
            82,
            85,
            3,
            98,
            81,
            37,
            6,
            55,
            38,
            74,
            52,
            13,
            77,
            80,
            83,
            20,
            60,
            97,
            41,
            14,
            49,
            29,
            54,
            58,
            30,
            21,
            39,
            61,
            75,
            90,
            31,
            35,
            63,
            12,
            70,
            47,
            76,
            92,
            78,
            53,
            11,
            15,
            88,
            62,
            68,
            19,
            4,
            66,
            2,
            67,
            22,
            86,
            5,
            17,
            25,
            46,
            10,
            73,
            71,
            36,
            48,
            1,
            16,
            50,
            26,
            44,
        ],
    )

    # Generate beeswarm plot
    shap.plots.beeswarm(
        shap_values,
        max_display=100,
        color=plt.get_cmap("RdYlBu_r"),
        order=feature_order[res],
    )
    plt.title("SHAP Beeswarm Plot")
    plt.savefig(f"output/ebm_uni_{res}_beeswarm_max100_order_v2.pdf", dpi=300, bbox_inches="tight")
    plt.clf()

    # Generate waterfall plot for the first example
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=100)
    plt.title("SHAP Waterfall Plot for Example 1")
    plt.savefig(f"output/ebm_uni_{res}_waterfall_v2.pdf", dpi=300, bbox_inches="tight")
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
        highres="data/Galaxy_uni_results_uni_highres_max-per-patient-1000_kmeans-100.xlsx",
    )

    for res, input_file_path in input_file_paths.items():
        main(input_file_path, res)
