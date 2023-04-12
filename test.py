import pandas as pd
from app import model_pred

new_data = {'Age': 68,
            'RestingBP': 150,
            'Cholesterol': 195,
            'Oldpeak': 0.0,
            'FastingBS': 1,
            'MaxHR': 132,
            }


def test_predict():
    features = pd.DataFrame(new_data)
    prediction = model_pred(features)
    assert prediction == 1
