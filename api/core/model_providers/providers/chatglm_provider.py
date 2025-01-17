import json
from json import JSONDecodeError
from typing import Type

import requests
from langchain.llms import ChatGLM

from core.helper import encrypter
from core.model_providers.models.base import BaseProviderModel
from core.model_providers.models.entity.model_params import ModelKwargsRules, KwargRule, ModelType, ModelMode
from core.model_providers.models.llm.chatglm_model import ChatGLMModel
from core.model_providers.providers.base import BaseModelProvider, CredentialsValidateFailedError
from models.provider import ProviderType


class ChatGLMProvider(BaseModelProvider):

    @property
    def provider_name(self):
        """
        Returns the name of a provider.
        """
        return 'chatglm'

    def _get_fixed_model_list(self, model_type: ModelType) -> list[dict]:
        if model_type == ModelType.TEXT_GENERATION:
            return [
                {
                    'id': 'chatglm3-6b',
                    'name': 'ChatGLM3-6B',
                    'mode': ModelMode.CHAT.value,
                },
                {
                    'id': 'chatglm3-6b-32k',
                    'name': 'ChatGLM3-6B-32K',
                    'mode': ModelMode.CHAT.value,
                },
                {
                    'id': 'chatglm2-6b',
                    'name': 'ChatGLM2-6B',
                    'mode': ModelMode.CHAT.value,
                }
            ]
        else:
            return []

    def _get_text_generation_model_mode(self, model_name) -> str:
        return ModelMode.CHAT.value

    def get_model_class(self, model_type: ModelType) -> Type[BaseProviderModel]:
        """
        Returns the model class.

        :param model_type:
        :return:
        """
        if model_type == ModelType.TEXT_GENERATION:
            model_class = ChatGLMModel
        else:
            raise NotImplementedError

        return model_class

    def get_model_parameter_rules(self, model_name: str, model_type: ModelType) -> ModelKwargsRules:
        """
        get model parameter rules.

        :param model_name:
        :param model_type:
        :return:
        """
        model_max_tokens = {
            'chatglm3-6b-32k': 32000,
            'chatglm3-6b': 8000,
            'chatglm2-6b': 8000,
        }

        max_tokens_alias = 'max_length' if model_name == 'chatglm2-6b' else 'max_tokens'

        return ModelKwargsRules(
            temperature=KwargRule[float](min=0, max=2, default=1, precision=2),
            top_p=KwargRule[float](min=0, max=1, default=0.7, precision=2),
            presence_penalty=KwargRule[float](enabled=False),
            frequency_penalty=KwargRule[float](enabled=False),
            max_tokens=KwargRule[int](alias=max_tokens_alias, min=10, max=model_max_tokens.get(model_name), default=2048, precision=0),
        )

    @classmethod
    def is_provider_credentials_valid_or_raise(cls, credentials: dict):
        """
        Validates the given credentials.
        """
        if 'api_base' not in credentials:
            raise CredentialsValidateFailedError('ChatGLM Endpoint URL must be provided.')

        try:
            response = requests.get(f"{credentials['api_base']}/v1/models", timeout=5)

            if response.status_code != 200:
                raise Exception('ChatGLM Endpoint URL is invalid.')
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    @classmethod
    def encrypt_provider_credentials(cls, tenant_id: str, credentials: dict) -> dict:
        credentials['api_base'] = encrypter.encrypt_token(tenant_id, credentials['api_base'])
        return credentials

    def get_provider_credentials(self, obfuscated: bool = False) -> dict:
        if self.provider.provider_type != ProviderType.CUSTOM.value:
            return {}
        try:
            credentials = json.loads(self.provider.encrypted_config)
        except JSONDecodeError:
            credentials = {
                'api_base': None
            }

        if credentials['api_base']:
            credentials['api_base'] = encrypter.decrypt_token(
                self.provider.tenant_id,
                credentials['api_base']
            )

            if obfuscated:
                credentials['api_base'] = encrypter.obfuscated_token(credentials['api_base'])

        return credentials

    @classmethod
    def is_model_credentials_valid_or_raise(cls, model_name: str, model_type: ModelType, credentials: dict):
        """
        check model credentials valid.

        :param model_name:
        :param model_type:
        :param credentials:
        """
        return

    @classmethod
    def encrypt_model_credentials(cls, tenant_id: str, model_name: str, model_type: ModelType,
                                  credentials: dict) -> dict:
        """
        encrypt model credentials for save.

        :param tenant_id:
        :param model_name:
        :param model_type:
        :param credentials:
        :return:
        """
        return {}

    def get_model_credentials(self, model_name: str, model_type: ModelType, obfuscated: bool = False) -> dict:
        """
        get credentials for llm use.

        :param model_name:
        :param model_type:
        :param obfuscated:
        :return:
        """
        return self.get_provider_credentials(obfuscated)
