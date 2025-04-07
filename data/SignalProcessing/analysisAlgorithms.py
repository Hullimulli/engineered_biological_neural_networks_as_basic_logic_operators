import numpy as np
from tqdm import tqdm
from statsmodels.nonparametric.kde import KDEUnivariate
def blahut_arimoto(p_y_x: np.ndarray,  log_base: float = 2, thresh: float = 1e-12, max_iter: int = 2*1e4, debug: bool = False) -> tuple:
    '''
    From https://github.com/kobybibas/blahut_arimoto_algorithm

    Maximize the capacity between I(X;Y)
    p_y_x: each row represents probability assignmnet P(Y|X)
    log_base: the base of the log when calculating the capacity
    thresh: the threshold of the update, finish the calculation when getting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert np.abs(p_y_x.sum(axis=1).mean() - 1) < 1e-6
    assert p_y_x.shape[0] > 1

    # The number of inputs: size of |X|
    m = p_y_x.shape[0]

    # The number of outputs: size of |Y|
    n = p_y_x.shape[1]

    # Initialize the prior uniformly
    r = np.ones((m,1)) / m

    # Compute the r(x) that maximizes the capacity
    for iteration in range(int(max_iter)):

        q = r * p_y_x
        q_norm = np.sum(q, axis=0, keepdims=True)
        q_norm[q_norm==0]=1
        q = q / q_norm

        r1 = np.prod(np.power(q, p_y_x), axis=1,keepdims=True)
        r1 = r1 / np.sum(r1)

        tolerance = np.linalg.norm(r1 - r)
        r = r1
        if tolerance < thresh:
            break

    # Calculate the capacity
    r = r.flatten()
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.nansum(r[i] * p_y_x[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    c = c / np.log(log_base)
    if debug:
        print(f"Stopped after {iteration+1} iterations. Last update was of size {tolerance}.")
    return c, r

def estimate_p_y_given_x(data, kernel='gau', bandwidth=0.025, num_query_points=250):
    """
    Estimates P(Y|X) using adaptive KDE for data of shape (|X|, n_samples).

    Args:
        data: Array of shape (|X|, n_samples), where each row corresponds to a discrete X
              and columns are continuous Y samples for that X.
        kernel: Kernel type for KDE (default: 'gau' for Gaussian).
        bandwidth: Bandwidth selection method or value (default: 'scott').
        num_query_points: Number of query points for Y (default: 250).

    Returns:
        query_points: Array of Y query points, shape (num_query_points, 1).
        p_y_given_x: Array of shape (|X|, num_query_points), with P(Y|X=x) for each X.
    """
    n_X, n_samples = data.shape

    # Determine query points dynamically
    data_min, data_max = np.min(data), np.max(data)
    query_points = np.linspace(data_min, data_max, num_query_points).reshape(-1, 1)

    # Initialize P(Y|X)
    p_y_given_x = np.zeros((n_X, num_query_points))

    # Estimate P(Y|X=x) using adaptive KDE
    for x in range(n_X):
        kde = KDEUnivariate(data[x, :])
        kde.fit(kernel=kernel, bw=((
                np.percentile(data, 99) - np.percentile(data, 1))) * bandwidth + 1e-16)  # Enable adaptive bandwidth
        p_y_given_x[x, :] = kde.evaluate(query_points.flatten())  # Evaluate on query points

    # Normalize densities so they sum to 1 for each X
    p_y_given_x = p_y_given_x / np.sum(p_y_given_x, axis=1, keepdims=True)

    return p_y_given_x
def calculate_MI_MAP(P_X, P_Y_given_X):
    # Compute P(X|Y)
    AP_values = P_Y_given_X * P_X[:, None]
    MAP_estimates = np.argmax(AP_values, axis=0)
    transition_matrix_map = np.zeros((len(P_X), len(P_X)))
    for input_idx in range(len(P_X)):
        for output_idx in range(len(P_X)):
            transition_matrix_map[input_idx, output_idx] = np.sum(P_Y_given_X[input_idx, MAP_estimates == output_idx])
    P_Y = P_X @ transition_matrix_map  # Compute P(Y)
    P_X_given_Y = transition_matrix_map * P_X[:, None] / (P_Y[None, :] + 1e-16)
    # Compute joint probability P(X, Y) = P(Y) * P(X|Y)
    P_XY = P_Y[None, :] * P_X_given_Y
    # Compute first term: sum_x,y P(X, Y) * log(P(X|Y))
    H_X_give_Y = -np.sum(P_XY * np.log2(P_X_given_Y + 1e-12))
    # Compute second term: sum_x P(X) * log(P(X))
    H_X = -np.sum(P_X * np.log2(P_X + 1e-12))
    return H_X - H_X_give_Y

def calculate_MI(P_X, transition_matrix):
    if len(P_X.shape) != 1:
        raise ValueError("P_X must be a 1D array representing the probability distribution of X.")
    if len(transition_matrix.shape) != 2:
        raise ValueError("transition_matrix must be a 2D array representing P(Y|X).")
    if P_X.shape[0] != transition_matrix.shape[0]:
        raise ValueError(
            f"The size of P_X ({P_X.shape[0]}) must match the number of rows in transition_matrix ({transition_matrix.shape[0]}).")
        # Check normalization
    if not np.isclose(P_X.sum(), 1):
        raise ValueError("P_X must sum to 1.")
    if not np.allclose(transition_matrix.sum(axis=1), 1):
        raise ValueError("Each row of transition_matrix must sum to 1 (valid conditional probabilities P(Y|X)).")
    # Compute P(X|Y)
    P_Y = P_X @ transition_matrix  # Compute P(Y)
    P_X_given_Y = transition_matrix * P_X[:, None] / (P_Y[None, :] + 1e-16)
    # Compute joint probability P(X, Y) = P(Y) * P(X|Y)
    P_XY = P_Y[None, :] * P_X_given_Y
    # Compute first term: sum_x,y P(X, Y) * log(P(X|Y))
    H_X_given_Y = -np.sum(P_XY * np.log2(P_X_given_Y + 1e-16))
    # Compute second term: sum_x P(X) * log(P(X))
    H_X = -np.sum(P_X * np.log2(P_X + 1e-16))
    return H_X - H_X_given_Y
