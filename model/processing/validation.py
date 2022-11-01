from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        MultiplePatientDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class PatientDataInputSchema(BaseModel):
    sex: Optional[str]
    age: Optional[int]
    children: Optional[int]
    smoker: Optional[str]
    region: Optional[str]
    bmi: Optional[float]


class MultiplePatientDataInputs(BaseModel):
    inputs: List[PatientDataInputSchema]
