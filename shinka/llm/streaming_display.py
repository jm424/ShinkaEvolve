"""
Inline streaming display for LLM thinking/reasoning traces.

This module provides a Rich-based terminal display that shows LLM
thinking and generation in real-time, inline with the existing log output.
"""

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from collections import deque
from typing import Optional, Literal
from dataclasses import dataclass
import threading
import time

ChunkType = Literal["thinking", "text", "waiting", "done"]


@dataclass
class StreamChunk:
    """A single chunk from the LLM stream."""
    chunk_type: ChunkType
    content: str
    timestamp: Optional[float] = None


class InlineThinkingDisplay:
    """
    A real-time inline display for LLM thinking/reasoning traces.
    
    Features:
    - Scrolling text viewport with configurable height
    - Visual distinction between thinking and text generation
    - Fading effect on older lines
    - Token/word count tracking
    - Thread-safe updates
    
    Usage:
        display = InlineThinkingDisplay(console)
        display.set_model("claude-opus-4-5")
        
        with display:
            # During streaming...
            display.add_chunk("thinking", "Analyzing the code...")
            display.add_chunk("text", "Here is my solution...")
        
        # After context manager exits, final state is shown
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        max_lines: int = 6,
        width: Optional[int] = None,
        show_thinking: bool = True,
        show_text: bool = False,
        fade_lines: int = 2,
    ):
        """
        Initialize the inline thinking display.
        
        Args:
            console: Rich Console instance (creates one if not provided)
            max_lines: Maximum number of lines to show in viewport
            width: Width of the panel (None = auto based on console width)
            show_thinking: Whether to display thinking chunks
            show_text: Whether to display text generation chunks
            fade_lines: Number of lines from top to apply fade effect
        """
        self.console = console or Console()
        self.max_lines = max_lines
        self.width = width
        self.show_thinking = show_thinking
        self.show_text = show_text
        self.fade_lines = fade_lines
        
        # State
        self._lines: deque = deque(maxlen=max_lines * 3)
        self._current_line = ""
        self._current_type: ChunkType = "waiting"
        self._model_name = ""
        self._start_time: Optional[float] = None
        self._token_count = 0
        self._thinking_tokens = 0
        self._text_tokens = 0
        
        # Threading
        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._live_started = False
        self._in_context = False
        
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def _get_panel_width(self) -> int:
        """Calculate panel width consistently."""
        if self.width is not None:
            return self.width
        return self.console.width - 2  # Leave small margin
    
    def _get_content_width(self) -> int:
        """Calculate usable content width inside the panel."""
        panel_width = self._get_panel_width()
        # Subtract: 2 for borders, 2 for padding, 4 for emoji prefix (emoji=2 chars + space)
        return panel_width - 8
    
    def _render(self) -> Panel:
        """Render the current state as a Rich Panel."""
        with self._lock:
            # Get visible lines
            visible = list(self._lines)[-self.max_lines:]
            
            # Add current incomplete line if exists
            if self._current_line.strip():
                visible.append((self._current_type, self._current_line))
            
            # Limit to max_lines
            visible = visible[-self.max_lines:]
        
        # Get content width for truncation
        content_width = self._get_content_width()
        
        # Build styled text
        text = Text()
        num_visible = len(visible)
        
        for i, (chunk_type, line) in enumerate(visible):
            # Calculate fade - older lines (at top) are more faded
            distance_from_bottom = num_visible - 1 - i
            is_faded = distance_from_bottom >= (num_visible - self.fade_lines)
            
            # Truncate lines to fit content width
            if len(line) > content_width:
                display_line = line[:content_width - 3] + "..."
            else:
                display_line = line
            
            if chunk_type == "thinking":
                if is_faded:
                    style = Style(color="cyan", italic=True, dim=True)
                else:
                    style = Style(color="cyan", italic=True)
                prefix = ">> "  # Use ASCII prefix to avoid emoji width issues
            else:  # "text"
                if is_faded:
                    style = Style(color="green", dim=True)  # Dim green, not white
                else:
                    style = Style(color="green")
                prefix = ""
            
            text.append(f"{prefix}{display_line}\n", style=style)
        
        # If no content yet, show placeholder (shouldn't normally happen with lazy init)
        if not visible:
            text.append(" \n", style="dim")  # Minimal placeholder
        
        # Determine panel styling based on current state
        if self._current_type == "thinking":
            border_style = "cyan"
            title = "[cyan]Thinking[/]"
        elif self._current_type == "text":
            border_style = "green"
            title = "[green]Generating[/]"
        elif self._current_type == "done":
            border_style = "dim green"
            title = "[dim green]Complete[/]"
        else:
            border_style = "dim"
            title = "[dim]Waiting[/]"
        
        # Build subtitle with stats
        model_short = self._model_name.split('/')[-1][-25:] if self._model_name else "..."
        
        elapsed = ""
        if self._start_time:
            elapsed_sec = time.time() - self._start_time
            elapsed = f"{elapsed_sec:.1f}s"
        
        stats_parts = [model_short]
        if self._thinking_tokens > 0:
            stats_parts.append(f"T:{self._thinking_tokens}")
        if self._text_tokens > 0:
            stats_parts.append(f"G:{self._text_tokens}")
        if elapsed:
            stats_parts.append(elapsed)
        
        subtitle = f"[dim]{' | '.join(stats_parts)}[/]"
        
        return Panel(
            text,
            title=title,
            subtitle=subtitle,
            border_style=border_style,
            width=self._get_panel_width(),
            padding=(0, 1),
        )
    
    def set_model(self, model_name: str):
        """Set the model name for display."""
        with self._lock:
            self._model_name = model_name
    
    def _start_live(self):
        """Start the Live display (called lazily on first chunk)."""
        if self._live_started or not self._in_context:
            return
        
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        self._live_started = True
    
    def add_chunk(self, chunk_type: ChunkType, content: str):
        """
        Add a streaming chunk to the display.
        
        Args:
            chunk_type: "thinking" or "text"
            content: The text content of the chunk
        """
        # Filter based on settings
        if chunk_type == "thinking" and not self.show_thinking:
            return
        if chunk_type == "text" and not self.show_text:
            return
        
        # Start live display on first chunk
        if not self._live_started:
            self._start_live()
        
        with self._lock:
            self._current_type = chunk_type
            
            # Track tokens
            tokens = self._estimate_tokens(content)
            self._token_count += tokens
            if chunk_type == "thinking":
                self._thinking_tokens += tokens
            else:
                self._text_tokens += tokens
            
            # Process content - handle newlines
            for char in content:
                if char == '\n':
                    if self._current_line.strip():  # Don't add empty lines
                        self._lines.append((chunk_type, self._current_line.strip()))
                    self._current_line = ""
                else:
                    self._current_line += char
        
        # Update display
        if self._live:
            try:
                self._live.update(self._render())
            except Exception:
                pass  # Ignore rendering errors
    
    def show_summary(self, thought: str, max_chars: int = 500):
        """
        Show a post-hoc summary of thinking (for non-streaming fallback).
        
        Args:
            thought: The complete thinking text
            max_chars: Maximum characters to show
        """
        if not thought:
            return
            
        with self._lock:
            self._current_type = "thinking"
            
            # Truncate if needed
            display_text = thought[:max_chars]
            if len(thought) > max_chars:
                display_text += "..."
            
            # Split into lines
            for line in display_text.split('\n'):
                if line.strip():
                    self._lines.append(("thinking", line.strip()))
            
            self._thinking_tokens = self._estimate_tokens(thought)
        
        # Show briefly
        if self._live:
            self._live.update(self._render())
    
    def finalize_thinking(self):
        """
        Finalize thinking phase - print thinking panel permanently and reset for generation.
        Call this after all thinking chunks have been added, before text generation starts.
        """
        with self._lock:
            # Only do something if we have thinking content
            if self._thinking_tokens == 0:
                return
            
            # Flush current line
            if self._current_line.strip():
                self._lines.append((self._current_type, self._current_line.strip()))
                self._current_line = ""
        
        # Stop current live display if running
        if self._live_started and self._live:
            self._live.__exit__(None, None, None)
            # Print thinking panel permanently
            self.console.print(self._render())
        
        # Reset for generation phase
        with self._lock:
            self._lines.clear()
            self._current_line = ""
            self._current_type = "waiting"
            self._live = None
            self._live_started = False
            # Keep token counts and timing for stats
    
    def finish(self):
        """Mark the display as complete."""
        with self._lock:
            self._current_type = "done"
            # Flush any remaining content
            if self._current_line.strip():
                self._lines.append((self._current_type, self._current_line.strip()))
                self._current_line = ""
        
        if self._live:
            self._live.update(self._render())
    
    def __enter__(self):
        """Context manager entry - mark we're in context (display starts lazily)."""
        self._start_time = time.time()
        self._in_context = True
        # Don't start Live yet - wait for first chunk
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finalize and stop live display."""
        self._in_context = False
        
        if self._live_started and self._live:
            self.finish()
            # Stop the live display (clears transient content)
            self._live.__exit__(exc_type, exc_val, exc_tb)
            # Print final state as permanent output
            self.console.print(self._render())
        
        return False  # Don't suppress exceptions

