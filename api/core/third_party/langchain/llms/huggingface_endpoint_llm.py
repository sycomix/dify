from typing import Dict, Any, Optional, List, Iterable, Iterator

from huggingface_hub import InferenceClient
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.huggingface_hub import VALID_TASKS
from langchain.llms import HuggingFaceEndpoint
from langchain.llms.utils import enforce_stop_tokens
from pydantic import root_validator

from langchain.utils import get_from_dict_or_env


class HuggingFaceEndpointLLM(HuggingFaceEndpoint):
    """HuggingFace Endpoint models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Only supports `text-generation` and `text2text-generation` for now.

    Example:
        .. code-block:: python

            from langchain.llms import HuggingFaceEndpoint
            endpoint_url = (
                "https://abcdefghijklmnop.us-east-1.aws.endpoints.huggingface.cloud"
            )
            hf = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token="my-api-key"
            )
    """
    client: Any
    streaming: bool = False

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )

        values['client'] = InferenceClient(values['endpoint_url'], token=huggingfacehub_api_token)

        values["huggingfacehub_api_token"] = huggingfacehub_api_token
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to HuggingFace Hub's inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = hf("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}

        # payload samples
        params = {**_model_kwargs, **kwargs}

        # generation parameter
        gen_kwargs = {
            **params,
            'stop_sequences': stop
        }

        response = self.client.text_generation(prompt, stream=self.streaming, details=True, **gen_kwargs)

        if self.streaming and isinstance(response, Iterable):
            combined_text_output = "".join(self._stream_response(response, run_manager))
            completion = combined_text_output
        else:
            completion = response.generated_text

        if self.task == "text-generation":
            text = completion
            # Remove prompt if included in generated text.
            if text.startswith(prompt):
                text = text[len(prompt) :]
        elif self.task == "text2text-generation":
            text = completion
        else:
            raise ValueError(
                f"Got invalid task {self.task}, "
                f"currently only {VALID_TASKS} are supported"
            )

        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)

        return text

    def _stream_response(
            self,
            response: Iterable,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Iterator[str]:
        for r in response:
            # skip special tokens
            if r.token.special:
                continue

            token = r.token.text
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=None
                )

            # yield the generated token
            yield token
