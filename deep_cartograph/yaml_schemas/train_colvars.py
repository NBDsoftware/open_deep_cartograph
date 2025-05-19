from pydantic import BaseModel
from typing import List, Union, Literal

class Optimizer(BaseModel):

    # Name of the optimizer (see torch.optim Algorithms)
    name: str = "Adam"
    # Keyword arguments for the optimizer (depends on the optimizer used, see torch.optim Algorithms)
    kwargs: dict = {'lr': 1.0e-03, 'weight_decay': 0.0}

class Architecture(BaseModel):

    # Fully connected hidden layers between the input and latent space
    hidden_layers: List[int] = [10, 10]
    # Lag time for TICA and DeepTICA
    lag_time: int = 1

class GeneralSettings(BaseModel):

    # Maximum number of tries for the training
    max_tries: int = 10
    # Seed for the PyTorch random number generator
    seed: int = 42
    # Lengths of the training and validation sets, e.g. [0.8, 0.2]
    lengths: List[float] = [0.8, 0.2]
    # Batch size for the training
    batch_size: int = 32
    # Maximum number of epochs for the training
    max_epochs: int = 1000
    # Dropout rate for the training
    dropout: float = 0.1
    # Shuffle the data before training
    shuffle: bool = False
    # Randomly split the data into training and validation sets
    random_split: bool = True
    # Do a validation check every n epochs
    check_val_every_n_epoch: int = 10
    # Save the model every n epochs
    save_check_every_n_epoch: int = 10

class InputColvars(BaseModel):
    
    # Start index to read the colvars file
    start: int = 0
    # Stop index to read the colvars file
    stop: Union[int, None] = None         # If None -> Read until the end
    # Step index to read the colvars file
    stride: int = 1

class EarlyStopping(BaseModel):

    # Patience for the early stopping, i.e., the number of validation checks with no improvement after which training will be stopped
    patience: int = 20
    # Minimum change in the loss function to consider it an improvement
    min_delta: float = 1.0e-05

class Trainings(BaseModel):
    
    # General settings
    general: GeneralSettings = GeneralSettings()
    # Early stopping settings
    early_stopping: EarlyStopping = EarlyStopping()
    # Optimizer settings
    optimizer: Optimizer = Optimizer()
    # Wether to save the training and validation losses after training
    save_loss: bool = True
    # Wether to plot the loss after training
    plot_loss: bool = True

class CommonCollectiveVariable(BaseModel):

    # Number of dimensions
    dimension: int = 2
    # Features normalization
    features_normalization: Literal['mean_std', 'min_max', 'none'] = 'mean_std'
    # Input colvars
    input_colvars: InputColvars = InputColvars()
    # Architecture settings (used only with NN-based Collective Variables)
    architecture: Architecture = Architecture()
    # Training settings (used only with NN-based Collective Variables)
    training: Trainings = Trainings()
    # Number of sub-spaces (used only with Hierarchical TICA)
    num_subspaces: int = 10
    # Dimension of the sub-spaces (used only with Hierarchical TICA)
    subspaces_dimension: int = 5

class FesFigure(BaseModel):
      
    # Calculate the Free Energy Surface
    compute: bool = True
    # Save the calculated Free Energy Surface in .npy files (otherwise it just plots 1D or 2D FES)
    save: bool = True
    # Temperature in Kelvin
    temperature: int = 300
    # Bandwidth for the Kernel Density Estimation of the Free Energy Surface
    bandwidth: float = 0.05
    # Number of bins for the Kernel Density Estimation of the Free Energy Surface
    num_bins: int = 150
    # Number of blocks for the standard error calculation of the Free Energy Surface
    num_blocks: int = 1
    # Maximum value for the Free Energy Surface (above which the value is set to NaN)
    max_fes: float = 30

class TrajProjection(BaseModel):
    
    # Plot the Projected Clustered Trajectory
    plot: bool = True
    # Number of bins for the Kernel Density Estimation of the Projected Clustered Trajectory
    num_bins: int = 100
    # Bandwidth for the Kernel Density Estimation of the Projected Clustered Trajectory
    bandwidth: float = 0.25
    # Transparency of the points in the Projected Clustered Trajectory
    alpha: float = 0.8
    # Colormap for the Projected Clustered Trajectory
    cmap: str = "turbo"
    # Use a legend in the Projected Clustered Trajectory plot
    use_legend: bool = True
    # Size of the markers in the Projected Clustered Trajectory
    marker_size: int = 5

class Figures(BaseModel):
      
    # Settings for the Free Energy Surface calculation
    fes: FesFigure = FesFigure()
    # Settings for the projection of the trajectory onto the CV space
    traj_projection: TrajProjection = TrajProjection()

class Clustering(BaseModel):

    # Note that:
    #  min_cluster_size should be set to the smallest size grouping that you wish to consider a cluster.
    #  the larger the value of min_samples you provide, the more conservative the clustering (more points will be declared as noise) and clusters will be restricted to progressively more dense areas

    # Whether to run the clustering or not
    run: bool = True
    # Clustering algorithm to use
    algorithm: Literal["kmeans", "hdbscan", "hierarchical"] = "hierarchical"
    # Whether to search for the optimal number of clusters inside the search_interval or not (only for hierarchical and kmeans)
    opt_num_clusters: bool = True
    # Range of number of clusters to search for the optimal number of clusters (only for hierarchical and kmeans)
    search_interval: List[int] = [3, 10]
    # Number of clusters to use (only for hierarchical and kmeans and if opt_num_clusters is false)
    num_clusters: int = 10
    # Linkage criterion to use ('ward', 'single', 'average', 'complete') (only for hierarchical)
    linkage: str = "complete"
    # Number of times the k-means algorithm is run with different centroid seeds (only for kmeans)
    n_init: int = 20
    # Minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise (only for hdbscan)
    min_cluster_size: int = 5
    # A limit to the size of clusters returned by the "eom" cluster selection algorithm, no limit if None (only for hdbscan)
    max_cluster_size: Union[int, None] = None
    # Number of samples in a neighborhood for a point to be considered as a core point (only for hdbscan)
    min_samples: int = 3
    # A distance threshold. Clusters below this value will be merged (only for hdbscan)
    cluster_selection_epsilon: float = 0
    # The method used to select clusters from the condensed tree."eom" selects the most persistent cluster while “leaf” provides the most fine grained and homogeneous ones
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    
class TrainColvarsSchema(BaseModel):
    
    # List of Collective Variables to train/calculate
    cvs: List[Literal['pca', 'ae', 'tica', 'htica', 'deep_tica']] = ['pca', 'ae', 'tica', 'htica', 'deep_tica']
    # Common settings for the Collective Variables
    common: CommonCollectiveVariable = CommonCollectiveVariable()
    # Settings for additional figures
    figures: Figures = Figures()
    # Settings for the clustering
    clustering: Clustering = Clustering()

    # Add Configuration class for this model
    class Config:
        # Allow extra fields - this is used to allow for model specific configurations that override common settings
        extra = "allow"

