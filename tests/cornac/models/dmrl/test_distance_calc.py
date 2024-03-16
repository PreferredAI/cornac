"""
Pytest tests for cornac.models.dmrl.d_cor_calc.py
"""

try:
    import torch
    import dcor
    from cornac.models.dmrl.d_cor_calc import DistanceCorrelationCalculator

    run_dmrl_test_funcs = True

except ImportError:
    run_dmrl_test_funcs = False


def skip_test_in_case_of_missing_reqs(test_func):
    test_func.__test__ = (
        run_dmrl_test_funcs  # Mark the test function as (non-)discoverable by unittest
    )
    return test_func


# first test the distance correlation calculator
@skip_test_in_case_of_missing_reqs
def test_distance_correlation_calculator():
    """
    Test the distance correlation calculator. Compare agains the library dcor.
    """
    num_neg = 4 + 1
    distance_cor_calc = DistanceCorrelationCalculator(n_factors=2, num_neg=num_neg)
    # create some fake data
    tensor_x = torch.randn(3, num_neg, 10)
    tensor_y = torch.randn(3, num_neg, 10)
    assert tensor_x.shape[1] == num_neg
    assert tensor_y.shape[1] == num_neg

    cor_per_sample = distance_cor_calc.calculate_cor(tensor_x, tensor_y)
    assert cor_per_sample.shape[0] == tensor_x.shape[1]

    for sample in range(num_neg - 1):
        # cutoff everyyhing after 5th decimal
        assert round(cor_per_sample[sample].item(), 2) == round(
            dcor.distance_correlation(
                tensor_x[:, sample, :], tensor_y[:, sample, :]
            ).item(),
            2,
        )
