from pydantic import BaseModel

from core.moderation.base import Moderation, ModerationInputsResult, ModerationOutputsResult, ModerationAction
from core.extension.api_based_extension_requestor import APIBasedExtensionRequestor, APIBasedExtensionPoint
from core.helper.encrypter import decrypt_token
from extensions.ext_database import db
from models.api_based_extension import APIBasedExtension


class ModerationInputParams(BaseModel):
    app_id: str = ""
    inputs: dict = {}
    query: str = ""


class ModerationOutputParams(BaseModel):
    app_id: str = ""
    text: str


class ApiModeration(Moderation):
    name: str = "api"

    @classmethod
    def validate_config(cls, tenant_id: str, config: dict) -> None:
        """
        Validate the incoming form config data.

        :param tenant_id: the id of workspace
        :param config: the form config data
        :return:
        """
        cls._validate_inputs_and_outputs_config(config, False)

        api_based_extension_id = config.get("api_based_extension_id")
        if not api_based_extension_id:
            raise ValueError("api_based_extension_id is required")

        extension = cls._get_api_based_extension(tenant_id, api_based_extension_id)
        if not extension:
            raise ValueError("API-based Extension not found. Please check it again.")

    def moderation_for_inputs(self, inputs: dict, query: str = "") -> ModerationInputsResult:
        flagged = False
        preset_response = ""

        if self.config['inputs_config']['enabled']:
            params = ModerationInputParams(
                app_id=self.app_id,
                inputs=inputs,
                query=query
            )

            result = self._get_config_by_requestor(APIBasedExtensionPoint.APP_MODERATION_INPUT, params.dict())
            return ModerationInputsResult(**result)

        return ModerationInputsResult(flagged=flagged, action=ModerationAction.DIRECT_OUTPUT, preset_response=preset_response)

    def moderation_for_outputs(self, text: str) -> ModerationOutputsResult:
        flagged = False
        preset_response = ""

        if self.config['outputs_config']['enabled']:
            params = ModerationOutputParams(
                app_id=self.app_id,
                text=text
            )

            result = self._get_config_by_requestor(APIBasedExtensionPoint.APP_MODERATION_OUTPUT, params.dict())
            return ModerationOutputsResult(**result)

        return ModerationOutputsResult(flagged=flagged, action=ModerationAction.DIRECT_OUTPUT, preset_response=preset_response)

    def _get_config_by_requestor(self, extension_point: APIBasedExtensionPoint, params: dict) -> dict:
        extension = self._get_api_based_extension(self.tenant_id, self.config.get("api_based_extension_id"))
        requestor = APIBasedExtensionRequestor(extension.api_endpoint, decrypt_token(self.tenant_id, extension.api_key))

        return requestor.request(extension_point, params)

    @staticmethod
    def _get_api_based_extension(tenant_id: str, api_based_extension_id: str) -> APIBasedExtension:
        return (
            db.session.query(APIBasedExtension)
            .filter(
                APIBasedExtension.tenant_id == tenant_id,
                APIBasedExtension.id == api_based_extension_id,
            )
            .first()
        )
