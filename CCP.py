import numpy as np
from scipy.optimize import minimize

# Function to estimate CCP using Hotz and Miller method
def estimate_ccp(data, covariates, num_choices, num_periods, discount_factor):
    num_obs = data.shape[0]
    num_covariates = covariates.shape[1]

    # Reshape data and covariates
    data_reshaped = data.reshape((num_obs, num_periods, num_choices))
    covariates_reshaped = covariates.reshape((num_obs, num_periods, num_covariates))

    # Initialize CCP matrix
    ccp_matrix = np.zeros((num_choices, num_covariates))

    for t in range(num_periods):
        # Get data and covariates for time period t
        choices_t = data_reshaped[:, t, :]
        covariates_t = covariates_reshaped[:, t, :]

        # Estimate CCP for each choice
        for i in range(num_choices):
            # Filter data for choice i
            choice_i_data = choices_t[:, i]
            choice_i_covariates = covariates_t[choice_i_data == 1, :]

            # Estimate CCP for choice i using Hotz and Miller method
            ccp_estimate = estimate_choice_prob(choice_i_covariates, covariates_t, discount_factor)

            # Update CCP matrix
            ccp_matrix[i, :] += ccp_estimate

    # Normalize CCP matrix
    ccp_matrix /= num_periods

    return ccp_matrix

# Function to estimate choice probabilities using Hotz and Miller method
def estimate_choice_prob(choice_data, covariates, discount_factor):
    num_obs = choice_data.shape[0]
    num_covariates = covariates.shape[1]

    # Estimate parameters using maximum likelihood
    initial_guess = np.zeros(num_covariates)
    result = minimize(negative_log_likelihood, initial_guess, args=(choice_data, covariates, discount_factor))
    params = result.x

    # Calculate choice probabilities
    choice_probs = np.exp(covariates @ params) / np.sum(np.exp(covariates @ params))

    return choice_probs

# Function to calculate negative log-likelihood
def negative_log_likelihood(params, choice_data, covariates, discount_factor):
    utilities = covariates @ params
    log_likelihood = np.sum(np.log(choice_data @ np.exp(utilities)))
    discount_factor_adjustment = (1 - discount_factor) * np.sum(utilities)
    return -log_likelihood + discount_factor_adjustment

# Example usage
data = np.array([[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 1],
                 [1, 0, 0]])

covariates = np.array([[1, 2],
                       [2, 1],
                       [3, 3],
                       [4, 2]])

num_choices = 2
num_periods = 1
discount_factor = 0.95

ccp_matrix = estimate_ccp(data, covariates, num_choices, num_periods, discount_factor)
print("Estimated CCP Matrix:")
print(ccp_matrix)
