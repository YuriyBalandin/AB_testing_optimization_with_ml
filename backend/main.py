from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.param_functions import File
import pandas as pd
import numpy as np
import io

from scipy import stats
from typing import List
import scipy.stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ttest_ind

from pycaret.regression import setup, compare_models, predict_model



app = FastAPI()

# Data validation
def validate_data(df):
    required_columns = ["ab_variant", "user_id", "metric_value"]

    # Check if required columns exist in the uploaded CSV
    if not all(column in df.columns for column in required_columns):
        raise ValueError("The CSV does not have the required columns")

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the uploaded data by splitting it based on the A/B variant.

    :param df: DataFrame containing the uploaded data.
    :return: Tuple of samples for control and variant groups.
    """
    # Split data based on A/B variant
    sample_a = df[df['ab_variant'] == 'control']['metric_value'].values
    sample_b = df[df['ab_variant'] == 'test']['metric_value'].values
    return sample_a, sample_b


def classic_bootstrap(sample_a, sample_b, func=np.mean, iterations=1000, alpha=0.05):
    """
    Applies the classic bootstrap method to compute the p-value, confidence interval,
    and difference of a statistic between two samples.

    :param sample_a: First sample of data.
    :param sample_b: Second sample of data.
    :param func: The function to compute the desired statistic (e.g., np.mean, np.median, etc.)
    :param iterations: Number of bootstrap iterations.
    :param alpha: Significance level for confidence intervals.
    :return: P-value, left bound of CI, right bound of CI, differences array.
    """
    len_a, len_b = len(sample_a), len(sample_b)
    diff_stats = []

    for _ in range(iterations):
        bootstrap_a = sample_a[np.random.choice(np.arange(len_a), len_a)]
        bootstrap_b = sample_b[np.random.choice(np.arange(len_b), len_b)]

        diff_stats.append(func(bootstrap_b) - func(bootstrap_a))

    delta_stat = func(sample_b) - func(sample_a)
    std_delta_stat = np.std(diff_stats)
    p_value_delta_stat = 2 * (1 - stats.norm.cdf(np.abs(delta_stat / std_delta_stat)))

    left_bound_stat = np.quantile(diff_stats, alpha / 2)
    right_bound_stat = np.quantile(diff_stats, 1 - alpha / 2)

    return p_value_delta_stat, left_bound_stat, right_bound_stat, np.array(diff_stats)


def plot_diff_stats(diff_stats, left_bound, right_bound, delta_stat, title="Bootstrapped Differences"):
    """
    Visualizes the bootstrapped differences, observed difference, and confidence interval bounds.

    :param diff_stats: Array of bootstrapped statistic differences.
    :param left_bound: Left bound of the confidence interval.
    :param right_bound: Right bound of the confidence interval.
    :param delta_stat: Observed difference in the statistic between the original samples.
    :param title: Title of the plot.
    """
    fig = px.histogram(diff_stats, nbins=100, title=title)
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=left_bound,
            x1=left_bound,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2),
        )
    )
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=right_bound,
            x1=right_bound,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2),
        )
    )
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=delta_stat,
            x1=delta_stat,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="blue", width=2),
        )
    )
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                line=dict(dash="dash", width=2),
                x0=left_bound,
                x1=left_bound,
                y0=0,
                y1=1,
                yref="paper",
            ),
            dict(
                type="line",
                line=dict(dash="dash", width=2),
                x0=right_bound,
                x1=right_bound,
                y0=0,
                y1=1,
                yref="paper",
            ),
            dict(
                type="line",
                line=dict(width=2),
                x0=delta_stat,
                x1=delta_stat,
                y0=0,
                y1=1,
                yref="paper",
            ),
        ]
    )
    fig.update_xaxes(title_text="Statistic Difference")
    fig.update_yaxes(title_text="Frequency")
    return fig


def plot_distributions(dist_1, dist_2, label_1='Distribution 1', label_2='Distribution 2', title='Distributions Comparison'):
    """
    Visualizes two distributions on the same plot.

    :param dist_1: np.array of the first distribution.
    :param dist_2: np.array of the second distribution.
    :param label_1: Label for the first distribution.
    :param label_2: Label for the second distribution.
    :param title: Title of the plot.
    """
    # Creating histogram trace for dist_1
    trace_1 = go.Histogram(
        x=dist_1,
        opacity=0.75,
        name=label_1,
        marker=dict(color='blue'),
        nbinsx=100
    )
    # Creating histogram trace for dist_2
    trace_2 = go.Histogram(
        x=dist_2,
        opacity=0.75,
        name=label_2,
        marker=dict(color='red'),
        nbinsx=100
    )
    # Defining data and layout, then plotting
    data = [trace_1, trace_2]
    layout = go.Layout(
        title=title,
        barmode='overlay',  # Overlay both histograms
        xaxis=dict(title='Value'),
        yaxis=dict(title='Frequency')
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def apply_CUPED(post_experiment_data, pre_experiment_data):
    """
    Apply the CUPED transformation to reduce variance in A/B testing results.

    :param post_experiment_data: DataFrame with columns ["ab_variant", "user_id", "metric_value"]
    :param pre_experiment_data: DataFrame with columns ["user_id", "metric_value"]
    :return: DataFrame with CUPED-transformed metric values
    """

    # Merge the pre-experiment data with the post-experiment data
    merged_data = pd.merge(post_experiment_data, pre_experiment_data, on='user_id', how='left', suffixes=('', '_pre'))

    # Fill NaN values with 0 for the pre-experiment metric value
    merged_data['metric_value_pre'].fillna(0, inplace=True)

    # Calculate the covariance and variance
    cov = np.cov(merged_data['metric_value'], merged_data['metric_value_pre'])[0, 1]
    var = np.var(merged_data['metric_value_pre'])

    # Compute the CUPED coefficient (theta)
    theta = cov / var

    # Adjust the post-experiment metric
    merged_data['metric_value'] = merged_data['metric_value'] - (merged_data['metric_value_pre'] - merged_data['metric_value_pre'].mean()) * theta

    # Return only the columns from the post-experiment data
    return merged_data[['ab_variant', 'user_id', 'metric_value']]




async def classic_bootstrap_analysis(df):
    """
    Performs classic bootstrap analysis on the processed data and generates necessary plots.

    :param processed_data: Tuple of samples for control and variant groups.
    :return: Dictionary containing the analysis results and plots.
    """
    sample_a, sample_b = preprocess_data(df)
    print(np.var(sample_a), np.var(sample_b))

    # Classic Bootstrap Analysis
    p_value, left_bound, right_bound, diff_stats = classic_bootstrap(sample_a, sample_b)

    # Creating the plots
    plot_1 = plot_distributions(sample_a, sample_b, 'Control', 'Variant', 'Initial Data Distribution')
    plot_2 = plot_diff_stats(diff_stats, left_bound, right_bound, np.mean(sample_b) - np.mean(sample_a))

    return {
        "initial_data_distribution": plot_1.to_json(),
        "bootstrapped_differences": plot_2.to_json(),
        "p_value": p_value,
        "confidence_interval": (left_bound, right_bound),
        "metric_value_control": np.mean(sample_a),
        "metric_value_variant": np.mean(sample_b),
        "message": "success result by classic_bootstrap_analysis"
    }




async def cuped_bootstrap_analysis(df, pre_experiment_data):
    """
    Performs CUPED transformation and classic bootstrap analysis on the processed data and generates necessary plots.

    :param processed_data: Tuple of samples for control and variant groups.
    :return: Dictionary containing the analysis results and plots.
    """
    # apply cuped
    transformed_data = apply_CUPED(df, pre_experiment_data)

    sample_a, sample_b = preprocess_data(transformed_data)
    print(np.var(sample_a), np.var(sample_b))

    # Classic Bootstrap Analysis
    p_value, left_bound, right_bound, diff_stats = classic_bootstrap(sample_a, sample_b)

    # Creating the plots
    plot_1 = plot_distributions(sample_a, sample_b, 'Control', 'Variant', 'Initial Data Distribution')
    plot_2 = plot_diff_stats(diff_stats, left_bound, right_bound, np.mean(sample_b) - np.mean(sample_a))

    return {
        "initial_data_distribution": plot_1.to_json(),
        "bootstrapped_differences": plot_2.to_json(),
        "p_value": p_value,
        "confidence_interval": (left_bound, right_bound),
        "metric_value_control": np.mean(sample_a),
        "metric_value_variant": np.mean(sample_b),
        "message": "success result by cuped_bootstrap_analysis"
    }




def train_best_model(df, target_col, session_id=42):
    """
    Trains the best model using PyCaret.

    :param df: DataFrame with features and target column.
    :param target_col: Name of the target column.
    :param session_id: Random seed for reproducibility.
    :return: Trained model.
    """
    # Setup PyCaret environment
    reg_setup = setup(data=df, target=target_col, session_id=session_id, verbose=False, html=False, n_jobs=-1)

    # Compare models and select the best one, excluding more complex models
    best_model = compare_models(exclude=['en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost'])
    return best_model


def apply_cupac(train_df, test_df, target_col, feature_cols):
    """
    Applies CUPAC on test data.

    :param train_df: Training DataFrame.
    :param test_df: Test DataFrame.
    :param target_col: Name of the target column.
    :param feature_cols: List of feature column names.
    :return: Test DataFrame with predictions and residuals added.
    """
    # Train the best model
    best_model = train_best_model(train_df[feature_cols + [target_col]], target_col)

    # Predict on test data
    test_df['predicted'] = predict_model(best_model, data=test_df[feature_cols])['prediction_label']

    # Compute residuals
    test_df['residual'] = test_df[target_col] - test_df['predicted']

    return test_df


async def cupac_bootstrap_analysis(df, pre_experiment_data):
    """
    Performs CUPAC transformation and classic bootstrap analysis on the A/B test data.

    :param df: DataFrame containing the A/B test data.
    :param pre_experiment_data: DataFrame containing the pre-experiment data for training the CUPAC model.
    :return: Dictionary containing the analysis results.
    """
    # Identifying feature columns in pre_experiment_data
    feature_cols = [col for col in pre_experiment_data.columns if col.startswith('feature_')]

    # Applying CUPAC
    df = apply_cupac(pre_experiment_data, df, 'metric_value', feature_cols)

    # Preprocess data for analysis
    sample_a = np.array(df[df['ab_variant'] == 'control']['residual'])
    sample_b = np.array(df[df['ab_variant'] == 'test']['residual'])
    print(np.var(sample_a), np.var(sample_b))
    
    # Classic Bootstrap Analysis
    p_value, left_bound, right_bound, diff_stats = classic_bootstrap(sample_a, sample_b)

    # Creating the plots
    plot_1 = plot_distributions(sample_a, sample_b, 'Control', 'Variant', 'Initial Data Distribution')
    plot_2 = plot_diff_stats(diff_stats, left_bound, right_bound, np.mean(sample_b) - np.mean(sample_a))

    return {
        "initial_data_distribution": plot_1.to_json(),
        "bootstrapped_differences": plot_2.to_json(),
        "p_value": p_value,
        "confidence_interval": (left_bound, right_bound),
        "metric_value_control": np.mean(sample_a),
        "metric_value_variant": np.mean(sample_b),
        "message": "success result by cuped_bootstrap_analysis"
    }



@app.post("/upload/")
async def upload_data(file: UploadFile = File(...), pre_experiment_file: UploadFile = None, method: str = Form(...)):
    # Debugging purpose
    print(f"Received method: {method}")

    # Read the uploaded file content and convert it into a DataFrame

    if method == "classic_bootstrap":

        file_content = await file.read()
        df = pd.read_csv(io.BytesIO(file_content))
        print(df.head())
    else:
        file_content = await file.read()
        df = pd.read_csv(io.BytesIO(file_content))

        pre_experiment_file_content = await pre_experiment_file.read()
        pre_experiment_data = pd.read_csv(io.BytesIO(pre_experiment_file_content))

    # Validate the data structure
    try:
        validate_data(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Analysis routing based on method
    if method == "classic_bootstrap":
        return await classic_bootstrap_analysis(df)
    elif method == "cuped_bootstrap":
        return await cuped_bootstrap_analysis(df, pre_experiment_data)
    elif method == "cupac_bootstrap":
        return await cupac_bootstrap_analysis(df, pre_experiment_data)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid analysis method selected: {method}")
