import numpy as np


def calculate_confidence_score(probs, type, normalize):
    if type == "min":
        return calculate_min_score(probs, normalize)
    elif type == "average":
        return calculate_average_score(probs, normalize)
    elif type == "average_low":
        return calculate_average_low_score(probs, normalize)
    elif type == "std":
        return calculate_std_score(probs, normalize)
    elif type == "std_low":
        return calculate_std_low_score(probs, normalize)
    elif type == "dot":
        return calculate_dot_score(probs, normalize)
    elif type == "max_entropy":
        return calculate_max_entropy_score(probs, normalize)
    elif type == "average_entropy":
        return calculate_average_entropy_score(probs, normalize)
    elif type == "perplexity":
        return calculate_perplexity_score(probs, normalize)
    elif type == "response_improbability":
        calculate_response_improbability_score(probs, normalize)
    elif type == "all":
        pass
    else:
        raise TypeError("No type found")


def calculate_normalized_score(scores):
    normalized_scores = []
    min_value = np.min(scores)
    max_value = np.max(scores)

    if max_value == min_value:
        # All values are the same, so we assign equal weight to all candidates
        normalized_scores = [1.0 for _ in scores]
    else:
        # Normalize the values between 0 and 1
        normalized_scores = [(v - min_value) / (max_value - min_value) for v in scores]
    
    return normalized_scores


def calculate_min_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        scores.append(min(candidate_probs))

    if normalize:
        scores = calculate_normalized_score(scores)

    return scores


def calculate_average_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        scores.append(sum(candidate_probs) / len(candidate_probs))

    if normalize:
        scores = calculate_normalized_score(scores)
    
    return scores


def calculate_average_low_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        filtered_candidate_probs = [item for item in candidate_probs if item != 1.0]
        if len(filtered_candidate_probs) != 0:
            scores.append(sum(filtered_candidate_probs) / len(filtered_candidate_probs))
        else:
            scores.append(1.0)
    
    if normalize:
        scores = calculate_normalized_score(scores)
    
    return scores


def calculate_std_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        scores.append(np.std(candidate_probs))

    if normalize:
        scores = calculate_normalized_score(scores)

    return scores


def calculate_std_low_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        filtered_candidate_probs = [item for item in candidate_probs if item != 1.0]
        if len(filtered_candidate_probs) != 0:
            scores.append(np.std(filtered_candidate_probs))
        else:
            scores.append(0.0)
    
    if normalize:
        scores = calculate_normalized_score(scores)
    
    return scores


def calculate_dot_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        product = 1
        for prob in candidate_probs:
            product *= prob
        n = len(candidate_probs)
        scores.append(product ** (1 / n))

    if normalize:
        scores = calculate_normalized_score(scores)
    
    return scores


def calculate_max_entropy_score(probs, normalize):
    pass


def calculate_average_entropy_score(probs, normalize):
    pass


def calculate_perplexity_score(probs, normalize):
    scores = []
    for candidate_probs in probs:
        log_probs = [np.log2(p) for p in candidate_probs]
        avg_log_prob = -np.mean(log_probs)
        perplexity = 2 ** avg_log_prob
        scores.append((1 / perplexity))
    
    if normalize:
        scores = calculate_normalized_score(scores)

    return scores


def calculate_response_improbability_score(probs, normalize):
    pass