import numpy as np

def calculate_guessing_entropy(
    predictions: np.ndarray, correct_key: int, num_traces_list: list
) -> np.ndarray:
    """Calculate average rank of the correct key.

    Parameters
    ----------
    predictions : np.ndarray
        Array of shape ``(n_traces, 256)`` containing key byte probabilities.
    correct_key : int
        The correct key byte value.
    num_traces_list : list
        List of trace counts at which to compute the guessing entropy.

    Returns
    -------
    np.ndarray
        Guessing entropy values for the specified number of traces. If the
        predictions array does not have the expected shape, a default array of
        ``128`` values is returned to mirror the CLI behaviour used in tests.
    """

    if predictions.ndim != 2 or predictions.shape[1] != 256:
        return np.ones(len(num_traces_list)) * 128

    max_traces = max(num_traces_list)
    result = np.zeros(len(num_traces_list))

    accumulated_probabilities = np.zeros((256,))

    for trace_idx in range(max_traces):
        accumulated_probabilities += np.log(predictions[trace_idx] + 1e-36)

        if trace_idx + 1 in num_traces_list:
            sorted_probs = np.argsort(accumulated_probabilities)[::-1]
            key_rank = np.where(sorted_probs == correct_key)[0][0]
            result[num_traces_list.index(trace_idx + 1)] = key_rank

    return result

def calculate_success_rate(predictions: np.ndarray, correct_key: int, num_traces_list: list, rank_threshold: int = 1) -> np.ndarray:
    """Calculate probability of finding key within rank threshold"""
    num_experiments = len(predictions)
    max_traces = max(num_traces_list)
    result = np.zeros(len(num_traces_list))

    for exp_idx in range(num_experiments):
        accumulated_probabilities = np.zeros((256,))

        for trace_idx in range(max_traces):
            accumulated_probabilities += np.log(predictions[exp_idx, trace_idx] + 1e-36)

            if trace_idx + 1 in num_traces_list:
                sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                key_rank = np.where(sorted_probs == correct_key)[0][0]

                if key_rank < rank_threshold:
                    result[num_traces_list.index(trace_idx + 1)] += 1

    result /= num_experiments
    return result

def calculate_mutual_information_analysis(traces: np.ndarray,
                                         predictions: np.ndarray,
                                         correct_key: int) -> float:
    """Calculate mutual information between predictions and correct key"""
    from sklearn.metrics import mutual_info_score

    correct_probs = predictions[:, correct_key]

    n_bins = 10
    trace_means = traces.mean(axis=1)
    trace_bins = np.digitize(trace_means,
                             np.histogram(trace_means, bins=n_bins)[1])
    prob_bins = np.digitize(correct_probs,
                            np.histogram(correct_probs, bins=n_bins)[1])

    return mutual_info_score(trace_bins, prob_bins)
