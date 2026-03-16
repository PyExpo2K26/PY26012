import numpy as np

def calculate_confidence(scores):
    score_values = np.array(list(scores.values()))
    agreement = 1.0 - (np.std(score_values) * 2.0)
    agreement = max(0.4, min(agreement, 1.0))
    return round(agreement * 100, 1)

def calculate_risk(scores, weights):
    final_weighted_score = sum(scores[k] * weights[k] for k in scores)
    return round(min(max(final_weighted_score, 0.0), 1.0) * 100, 1)

# Test cases
weights = {
    'cnn':       0.85,
    'frequency': 0.05,
    'noise':     0.05,
    'ela':       0.05,
    'copymove':  0.00,
    'metadata':  0.00,
}

test_scenarios = [
    {
        "name": "High Agreement (Deepfake)",
        "scores": {'cnn': 0.9, 'frequency': 0.85, 'noise': 0.8, 'ela': 0.75, 'copymove': 0.1, 'metadata': 0.0}
    },
    {
        "name": "Mixed Signal (Inconsistent)",
        "scores": {'cnn': 0.9, 'frequency': 0.1, 'noise': 0.1, 'ela': 0.1, 'copymove': 0.1, 'metadata': 0.0}
    },
    {
        "name": "Clean Image (Natural)",
        "scores": {'cnn': 0.1, 'frequency': 0.05, 'noise': 0.1, 'ela': 0.05, 'copymove': 0.0, 'metadata': 0.0}
    }
]

print("--- Scoring Logic Verification ---")
for scenario in test_scenarios:
    risk = calculate_risk(scenario['scores'], weights)
    conf = calculate_confidence(scenario['scores'])
    print(f"Scenario: {scenario['name']}")
    print(f"  Risk: {risk}%")
    print(f"  Confidence: {conf}%")
    print("-" * 30)

# Expected behavior: 
# High Agreement should have higher confidence than Mixed Signal.
# Clean Image should also have relatively high confidence.

# Verification of weight sum
total_weight = sum(weights.values())
print(f"Total weight sum: {total_weight} (Should be 1.0)")
