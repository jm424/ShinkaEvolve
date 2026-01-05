import backoff
import anthropic
from typing import Optional, TYPE_CHECKING
from .pricing import CLAUDE_MODELS
from .result import QueryResult
import logging

if TYPE_CHECKING:
    from shinka.llm.streaming_display import InlineThinkingDisplay

logger = logging.getLogger(__name__)


MAX_TRIES = 20
MAX_VALUE = 20


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Anthropic - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
    ),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    on_backoff=backoff_handler,
)
def query_anthropic(
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
    """Query Anthropic/Bedrock model."""
    new_msg_history = msg_history + [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": msg,
                }
            ],
        }
    ]

    # Models that require streaming for long operations (>10 minutes)
    REQUIRES_STREAMING = ["claude-opus-4-5-20251101"]
    # Use streaming if model requires it OR if we have a display to stream to
    use_streaming = model in REQUIRES_STREAMING or streaming_display is not None

    if output_model is None:
        if use_streaming:
            # Use streaming for models that require it or when display is requested
            thought = ""
            content = ""
            input_tokens = 0
            output_tokens = 0

            # Set model name on display if provided
            if streaming_display:
                streaming_display.set_model(model)

            with client.messages.stream(
                model=model,
                system=system_msg,
                messages=new_msg_history,
                **kwargs,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            # Check if this is a thinking block
                            if hasattr(event, 'content_block') and hasattr(event.content_block, 'type'):
                                if event.content_block.type == 'thinking':
                                    pass  # We'll collect thinking content in delta events
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                if hasattr(event.delta, 'type'):
                                    if event.delta.type == 'thinking_delta':
                                        thought += event.delta.thinking
                                        # Stream to display
                                        if streaming_display:
                                            streaming_display.add_chunk("thinking", event.delta.thinking)
                                    elif event.delta.type == 'text_delta':
                                        content += event.delta.text
                                        # Stream to display
                                        if streaming_display:
                                            streaming_display.add_chunk("text", event.delta.text)
                        elif event.type == 'message_start':
                            if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                                input_tokens = event.message.usage.input_tokens
                        elif event.type == 'message_delta':
                            if hasattr(event, 'usage'):
                                output_tokens = event.usage.output_tokens

                # Get final message to ensure we have usage stats
                final_message = stream.get_final_message()
                if hasattr(final_message, 'usage'):
                    input_tokens = final_message.usage.input_tokens
                    output_tokens = final_message.usage.output_tokens
        else:
            # Use non-streaming for other models
            response = client.messages.create(
                model=model,
                system=system_msg,
                messages=new_msg_history,
                **kwargs,
            )
            # Separate thinking from non-thinking content
            if len(response.content) == 1:
                thought = ""
                content = response.content[0].text
            else:
                thought = response.content[0].thinking
                content = response.content[1].text

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
    else:
        raise NotImplementedError("Structured output not supported for Anthropic.")

    new_msg_history.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }
    )
    input_cost = CLAUDE_MODELS[model]["input_price"] * input_tokens
    output_cost = CLAUDE_MODELS[model]["output_price"] * output_tokens
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
