import backoff
import openai
import re
from typing import Optional, TYPE_CHECKING
from .pricing import GEMINI_MODELS
from .result import QueryResult
import logging

if TYPE_CHECKING:
    from shinka.llm.streaming_display import InlineThinkingDisplay

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
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
def query_gemini(
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
    """Query Gemini model via OpenAI-compatible API."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    # Determine if we should use streaming
    use_streaming = streaming_display is not None
    
    if output_model is not None:
        raise ValueError("Gemini does not support structured output.")
    
    # Set model name on display if provided
    if streaming_display:
        streaming_display.set_model(model)
    
    if use_streaming:
        # Streaming path
        thought = ""
        content = ""
        input_tokens = 0
        output_tokens = 0
        
        # Create streaming request
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            stream=True,
            **kwargs,
        )
        
        for chunk in stream:
            # Handle different chunk structures
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            choice = chunk.choices[0]
            
            # Check for thinking/reasoning content
            # Gemini 3 with thinking_config may return reasoning in different fields
            reasoning_content = None
            is_thinking = False
            
            # Check various possible locations for thought summaries
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content = delta.reasoning_content
                is_thinking = True
            elif hasattr(delta, 'reasoning') and delta.reasoning:
                reasoning_content = delta.reasoning
                is_thinking = True
            
            # Check extra_content for Gemini 3's thought flag
            # Gemini 3 returns: delta.extra_content = {'google': {'thought': True}}
            if hasattr(delta, 'extra_content') and delta.extra_content:
                google_extra = delta.extra_content.get('google', {}) if isinstance(delta.extra_content, dict) else {}
                if google_extra.get('thought'):
                    is_thinking = True
            
            # Check finish_reason for thinking indicator
            if hasattr(choice, 'finish_reason') and choice.finish_reason == 'thinking':
                is_thinking = True
            
            if reasoning_content:
                thought += reasoning_content
                if streaming_display:
                    streaming_display.add_chunk("thinking", reasoning_content)
            
            # Regular content - check if it's actually thinking content
            if delta.content:
                if is_thinking:
                    thought += delta.content
                    if streaming_display:
                        streaming_display.add_chunk("thinking", delta.content)
                else:
                    content += delta.content
                    if streaming_display:
                        streaming_display.add_chunk("text", delta.content)
            
            # Try to capture usage from chunk if available
            if hasattr(chunk, 'usage') and chunk.usage:
                if hasattr(chunk.usage, 'prompt_tokens'):
                    input_tokens = chunk.usage.prompt_tokens
                if hasattr(chunk.usage, 'completion_tokens'):
                    output_tokens = chunk.usage.completion_tokens
                elif hasattr(chunk.usage, 'total_tokens'):
                    output_tokens = chunk.usage.total_tokens - input_tokens
        
        # For streaming, we may not have exact token counts
        # Estimate if not available
        if input_tokens == 0:
            # Rough estimate: 4 chars per token
            input_tokens = len(msg + system_msg) // 4
        if output_tokens == 0:
            output_tokens = len(content + thought) // 4
        
        text = content
        
    else:
        # Non-streaming path (original implementation)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            text = response.choices[0].message.content
        except Exception:
            # Reasoning models - ResponseOutputMessage
            text = response.output[1].content[0].text
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
        
        # Extract thought from <thought> tags (legacy approach)
        thought_match = re.search(
            r"<thought>(.*?)</thought>", text, re.DOTALL
        )
        thought = thought_match.group(1) if thought_match else ""
        
        # Remove thought tags from content
        content_match = re.search(
            r"<thought>(.*?)</thought>", text, re.DOTALL
        )
        if content_match:
            content = (
                text[: content_match.start()]
                + text[content_match.end() :]
            ).strip()
        else:
            content = text
        
        # Show thought post-hoc if display is provided but we didn't stream
        # (This shouldn't happen with current logic, but kept for safety)
        if streaming_display and thought:
            streaming_display.show_summary(thought)
    
    new_msg_history.append({"role": "assistant", "content": text})
    
    input_cost = GEMINI_MODELS[model]["input_price"] * input_tokens
    output_cost = GEMINI_MODELS[model]["output_price"] * output_tokens
    
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
