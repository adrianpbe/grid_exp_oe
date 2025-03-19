import gymnasium as gym

from grid_exp_oe.models.base import ModelBuilder, ModelHparams, PolicyType
from grid_exp_oe.models.conv_actor_critic import ConvActorCriticBuilder
from grid_exp_oe.models.lstm_conv_actor_critic import LSTMConvActorCriticBuilder


AVAILABLE_MODELS_BUILDERS: dict[str, ModelBuilder] = {
    ConvActorCriticBuilder.model_id(): ConvActorCriticBuilder,
    LSTMConvActorCriticBuilder.model_id(): LSTMConvActorCriticBuilder,
}


def available_models() -> list[str]:
    return list(AVAILABLE_MODELS_BUILDERS.keys())


def get_model_builder_cls(model_id: str) -> type[ModelBuilder]:
    try:
        Builder = AVAILABLE_MODELS_BUILDERS[model_id]
    except KeyError:
        raise ValueError(f"model with mdoel_id {model_id} not found!")
    return Builder


def get_model_builder(model_id: str, model_config_data: dict | None) -> ModelBuilder:
    model_config_data = {} if model_config_data is None else model_config_data
    ModelBuilderCls = get_model_builder_cls(model_id)
    HparamsCls = ModelBuilderCls.HParams()
    model_hparams = HparamsCls(**model_config_data)
    return ModelBuilderCls(model_hparams)
