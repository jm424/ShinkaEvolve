import backoff
import openai
from typing import Optional, TYPE_CHECKING
from .pricing import DEEPSEEK_MODELS
from .result import QueryResult
import logging

if TYPE_CHECKING:
    from shinka.llm.streaming_display import InlineThinkingDisplay

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"DeepSeek - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=5,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_deepseek(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    streaming_display: Optional["InlineThinkingDisplay"] = None,
    **kwargs,
) -> QueryResult:
    """Query DeepSeek model via OpenAI-compatible API."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for DeepSeek.")
    
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Determine if we should use streaming
    use_streaming = streaming_display is not None
    
    # Set model name on display if provided
    if streaming_display:
        streaming_display.set_model(model)
    
    if use_streaming:
        # Streaming path
        thought = ""
        content = ""
        input_tokens = 0
        output_tokens = 0
        
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            stream=True,
            n=1,
            stop=None,
            **kwargs,
        )
        
        for chunk in stream:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # Check for reasoning content (DeepSeek R1 models)
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                thought += delta.reasoning_content
                if streaming_display:
                    streaming_display.add_chunk("thinking", delta.reasoning_content)
            
            # Regular content
            if delta.content:
                content += delta.content
                if streaming_display:
                    streaming_display.add_chunk("text", delta.content)
            
            # Try to capture usage if available
            if hasattr(chunk, 'usage') and chunk.usage:
                if hasattr(chunk.usage, 'prompt_tokens'):
                    input_tokens = chunk.usage.prompt_tokens
                if hasattr(chunk.usage, 'completion_tokens'):
                    output_tokens = chunk.usage.completion_tokens
        
        # Estimate tokens if not available
        if input_tokens == 0:
            input_tokens = len(msg + system_msg) // 4
        if output_tokens == 0:
            output_tokens = len(content + thought) // 4
    
    else:
        # Non-streaming path (original implementation)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            n=1,
            stop=None,
            **kwargs,
        )
        content = response.choices[0].message.content
        try:
            thought = response.choices[0].message.reasoning_content
        except Exception:
            thought = ""
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    
    new_msg_history.append({"role": "assistant", "content": content})
    
    input_cost = DEEPSEEK_MODELS[model]["input_price"] * input_tokens
    output_cost = DEEPSEEK_MODELS[model]["output_price"] * output_tokens
    
    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
