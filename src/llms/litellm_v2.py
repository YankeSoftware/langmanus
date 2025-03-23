from langchain_community.chat_models import ChatLiteLLM

from operator import itemgetter
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.language_models import LanguageModelInput


from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    is_basemodel_subclass,
)
from pydantic import BaseModel
import logging
import re

logger = logging.getLogger(__name__)

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


class ChatLiteLLMV2(ChatLiteLLM):
    def _is_mistral_model(self, model_name: str) -> bool:
        """
        Check if the model is a Mistral model that requires special handling.
        """
        if not model_name:
            return False
        
        model_lower = model_name.lower()
        
        # Direct Mistral model checks
        if any(mistral_id in model_lower for mistral_id in ['mistral', 'istral', 'mixtral']):
            return True
        
        # Check for common Mistral quantized models
        quantized_patterns = [
            r'mistral.*7b',
            r'mixtral.*8x7b',
            r'mistral.*instruct',
            r'zephyr',  # Zephyr is based on Mistral
            r'openhermes',  # OpenHermes uses Mistral architecture
        ]
        
        for pattern in quantized_patterns:
            if re.search(pattern, model_lower):
                return True
            
        return False

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        logger.debug(f"with_structured_output called with schema={schema.__name__ if _is_pydantic_class(schema) else schema}, method={method}")
        logger.debug(f"Additional kwargs: {kwargs}")
        
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if strict is not None and method == "json_mode":
            raise ValueError(
                "Argument `strict` is not supported with `method`='json_mode'"
            )
        is_pydantic_schema = _is_pydantic_class(schema)
        logger.debug(f"Is pydantic schema: {is_pydantic_schema}")

        # Check model information
        model_name = self.model_name.lower() if hasattr(self, "model_name") else ""
        logger.debug(f"Model name: {model_name}")
        
        is_mistral = self._is_mistral_model(model_name)
        logger.debug(f"Is Mistral model: {is_mistral}")
        
        # Enhanced detection for LM Studio
        lm_studio_mode = False
        api_base = None
        
        try:
            # Log all attributes to help debugging
            if hasattr(self, "api_base"):
                api_base = self.api_base
                logger.debug(f"API Base URL: {api_base}")
                
                # Verify server availability early
                if api_base and ("localhost" in api_base or "127.0.0.1" in api_base):
                    import requests
                    try:
                        response = requests.get(f"{api_base}/models", timeout=2)
                        if response.status_code != 200:
                            logger.warning(f"LM Studio API returned status code {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error connecting to LM Studio API at {api_base}: {e}")
                        logger.error("Please ensure LM Studio is running with the API server started")
            else:
                logger.debug("No api_base attribute found")
            
            # Special handling for LM Studio models which use different format
            if hasattr(self, "api_base") and self.api_base:
                if "lmstudio" in self.api_base.lower():
                    lm_studio_mode = True
                elif "localhost" in self.api_base.lower() or "127.0.0.1" in self.api_base.lower():
                    lm_studio_mode = True
            
            logger.debug(f"LM Studio mode: {lm_studio_mode}, Model: {model_name}, Is Mistral: {is_mistral}, Method: {method}")

            if method == "function_calling":
                if schema is None:
                    raise ValueError(
                        "schema must be specified when method is not 'json_mode'. "
                        "Received None."
                    )
                tool_name = convert_to_openai_tool(schema)["function"]["name"]
                
                # LM Studio with Mistral models doesn't support function/tool calling directly
                if lm_studio_mode and is_mistral:
                    logger.info("Using manual function calling for LM Studio with Mistral model")
                    # Fall back to manual approach
                    return self._create_manual_json_output_chain(schema, include_raw)
                    
                bind_kwargs = self._filter_disabled_params(
                    tool_choice=tool_name,
                    parallel_tool_calls=False,
                    strict=strict,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )

                llm = self.bind_tools([schema], **bind_kwargs)
                if is_pydantic_schema:
                    output_parser: Runnable = PydanticToolsParser(
                        tools=[schema],  # type: ignore[list-item]
                        first_tool_only=True,  # type: ignore[list-item]
                    )
                else:
                    output_parser = JsonOutputKeyToolsParser(
                        key_name=tool_name, first_tool_only=True
                    )
            elif method == "json_mode":
                # For LM Studio compatibility
                if lm_studio_mode and is_mistral:
                    logger.info("Using manual JSON mode for LM Studio with Mistral model")
                    # Fall back to manual approach for Mistral in LM Studio
                    return self._create_manual_json_output_chain(schema, include_raw)
                    
                # Standard approach for models that support json_mode
                response_format_type = "json_schema" if lm_studio_mode else "json_object"
                llm = self.bind(
                    response_format={"type": response_format_type},
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )
                output_parser = (
                    PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                    if is_pydantic_schema
                    else JsonOutputParser()
                )
            else:
                raise ValueError(
                    f"Unrecognized method argument. Expected one of 'function_calling' or "
                    f"'json_mode'. Received: '{method}'"
                )

            if include_raw:
                parser_assign = RunnablePassthrough.assign(
                    parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
                )
                parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                parser_with_fallback = parser_assign.with_fallbacks(
                    [parser_none], exception_key="parsing_error"
                )
                return RunnableMap(raw=llm) | parser_with_fallback
            else:
                return llm | output_parser
        except Exception as e:
            logger.error(f"Error in with_structured_output: {e}")
            raise
            
    def _create_manual_json_output_chain(
        self, 
        schema: Optional[_DictOrPydanticClass] = None,
        include_raw: bool = False
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """
        Create a manual chain for JSON output when model doesn't support native JSON output modes.
        This adds instructions to the prompt and manually parses the result.
        """
        from src.utils.json_utils import parse_structured_output
        
        # Define a runnable that adds JSON instructions to the prompt
        def add_json_instructions(inputs):
            if isinstance(inputs, list):
                # Add JSON instruction to the end of the list
                enhanced_inputs = inputs.copy()
                schema_desc = f"{schema.__name__}" if _is_pydantic_class(schema) else "dictionary"
                enhanced_inputs.append({
                    "role": "user",
                    "content": f"Respond ONLY with valid JSON matching the {schema_desc} schema. No text before or after."
                })
                return enhanced_inputs
            return inputs
        
        # Define a parser that extracts JSON from the response
        def manual_json_parser(response):
            parsed = parse_structured_output(response, schema if _is_pydantic_class(schema) else None)
            return parsed
        
        # Build the chain
        json_chain = (
            RunnablePassthrough.assign(enhanced=add_json_instructions) 
            | RunnablePassthrough.assign(
                raw=lambda x: self.invoke(x["enhanced"]),
            )
            | RunnablePassthrough.assign(
                parsed=lambda x: manual_json_parser(x["raw"])
            )
        )
        
        # Return the appropriate output based on include_raw
        if include_raw:
            return RunnableMap(
                raw=lambda x: x["raw"],
                parsed=lambda x: x["parsed"]
            ) | json_chain
        else:
            return json_chain | (lambda x: x["parsed"])
