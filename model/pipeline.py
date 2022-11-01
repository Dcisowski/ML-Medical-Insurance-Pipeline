from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from model.config.core import config
from model.processing import features as pp

patient_pipe = Pipeline(
    [
        # ==== VARIABLE TRANSFORMATION =====
        (
            "yeo",
            YeoJohnsonTransformer(variables=config.model_config.numericals_yeo_vars),
        ),
        # === mappers ===
        (
            "mapper_sex",
            pp.Mapper(
                variables=config.model_config.sex_vars,
                mappings=config.model_config.sex_mappings,
            ),
        ),
        (
            "mapper_smoker",
            pp.Mapper(
                variables=config.model_config.smoker_vars,
                mappings=config.model_config.smoker_mappings,
            ),
        ),
        (
            "mapper_region",
            pp.Mapper(
                variables=config.model_config.region_vars,
                mappings=config.model_config.region_mappings,
            ),
        ),
        ("scaler", MinMaxScaler()),
        (
            "Lasso",
            Lasso(
                alpha=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
