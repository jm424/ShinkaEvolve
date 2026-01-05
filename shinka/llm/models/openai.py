import backoff
import openai
from typing import Optional, TYPE_CHECKING
from .pricing import OPENAI_MODELS
from .result import QueryResult
import logging

if TYPE_CHECKING:
    from shinka.llm.streaming_display import InlineThinkingDisplay

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenAI - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_openai(
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
    """Query OpenAI model via Responses API."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Determine if we should use streaming
    use_streaming = streaming_display is not None
    
    # Set model name on display if provided
    if streaming_display:
        streaming_display.set_model(model)
    
    if output_model is None:
        if use_streaming:
            # Streaming path for Responses API
            thought = ""
            content = ""
            input_tokens = 0
            output_tokens = 0
            current_output_type = "message"  # Track reasoning vs message
            
            # Remove 'stream' from kwargs if present to avoid conflict
            stream_kwargs = {k: v for k, v in kwargs.items() if k != 'stream'}
            
            # Add reasoning.summary parameter if reasoning is present
            if 'reasoning' in stream_kwargs and isinstance(stream_kwargs['reasoning'], dict):
                stream_kwargs['reasoning']['summary'] = 'detailed'
            
            stream = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                stream=True,
                **stream_kwargs,
            )
            
            for event in stream:
                event_type = getattr(event, 'type', None)
                
                # Track if current output is reasoning or message
                if event_type == 'response.output_item.added':
                    item = getattr(event, 'item', None)
                    if item:
                        item_type = getattr(item, 'type', None)
                        if item_type == 'reasoning':
                            current_output_type = 'reasoning'
                        elif item_type == 'message':
                            current_output_type = 'message'
                
                # Handle output item completion - capture reasoning content/summary
                elif event_type == 'response.output_item.done':
                    item = getattr(event, 'item', None)
                    if item:
                        item_type = getattr(item, 'type', None)
                        if item_type == 'reasoning':
                            # Extract reasoning summary (OpenAI provides this post-hoc, not streamed)
                            # Try to extract from summary array
                            summary = getattr(item, 'summary', []) or []
                            for summary_item in summary:
                                summary_text = getattr(summary_item, 'text', '')
                                if summary_text:
                                    thought += summary_text + "\n"
                                    if streaming_display:
                                        streaming_display.add_chunk("thinking", summary_text + "\n")
                            
                            # Try to extract from content array
                            content_arr = getattr(item, 'content', []) or []
                            for content_item in content_arr:
                                content_text = getattr(content_item, 'text', '')
                                if content_text:
                                    thought += content_text + "\n"
                                    if streaming_display:
                                        streaming_display.add_chunk("thinking", content_text + "\n")
                            
                            # Finalize thinking display - print it permanently before generation starts
                            if streaming_display:
                                streaming_display.finalize_thinking()
                
                # Handle text deltas
                elif event_type in ('response.output_text.delta', 'response.text.delta'):
                    delta_text = getattr(event, 'delta', '')
                    if delta_text:
                        if current_output_type == 'reasoning':
                            thought += delta_text
                            if streaming_display:
                                streaming_display.add_chunk("thinking", delta_text)
                        else:
                            content += delta_text
                            if streaming_display:
                                streaming_display.add_chunk("text", delta_text)
                
                # Handle content part deltas
                elif event_type == 'response.content_part.delta':
                    delta = getattr(event, 'delta', None)
                    if delta:
                        delta_text = getattr(delta, 'text', '') or str(delta)
                        if delta_text:
                            if current_output_type == 'reasoning':
                                thought += delta_text
                                if streaming_display:
                                    streaming_display.add_chunk("thinking", delta_text)
                            else:
                                content += delta_text
                                if streaming_display:
                                    streaming_display.add_chunk("text", delta_text)
                
                # Handle completion
                elif event_type in ('response.done', 'response.completed'):
                    response_obj = getattr(event, 'response', None)
                    if response_obj and hasattr(response_obj, 'usage'):
                        input_tokens = response_obj.usage.input_tokens
                        output_tokens = response_obj.usage.output_tokens
                    break
            
            # Estimate tokens if not available
            if input_tokens == 0:
                input_tokens = len(msg + system_msg) // 4
            if output_tokens == 0:
                output_tokens = len(content + thought) // 4
            
        else:
            # Non-streaming path (original implementation)
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_msg},
                    *new_msg_history,
                ],
                **kwargs,
            )
            try:
                content = response.output[0].content[0].text
            except Exception:
                # Reasoning models - ResponseOutputMessage
                content = response.output[1].content[0].text
            thought = ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        
        new_msg_history.append({"role": "assistant", "content": content})
        
    else:
        # Structured output - no streaming support
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            text_format=output_model,
            **kwargs,
        )
        content = response.output_parsed
        new_content = ""
        for i in content:
            new_content += i[0] + ":" + i[1] + "\n"
        new_msg_history.append({"role": "assistant", "content": new_content})
        thought = ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    input_cost = OPENAI_MODELS[model]["input_price"] * input_tokens
    output_cost = OPENAI_MODELS[model]["output_price"] * output_tokens
    
    result = QueryResult(
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
    return result
