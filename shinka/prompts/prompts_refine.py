"""Refinement prompts for iterative candidate improvement.

These prompts are used when a candidate program needs refinement based on
evaluation feedback (compilation errors, correctness failures, performance data).
"""

# =============================================================================
# DIFF-BASED REFINEMENT (token-efficient, for targeted fixes)
# =============================================================================

REFINE_DIFF_SYSTEM_MSG = """You are refining optimized code that was previously generated and evaluated.
Your task is to fix the specific issues identified in the evaluation feedback while PRESERVING the optimization strategy.

IMPORTANT: Do NOT revert optimizations or replace custom kernels with vanilla PyTorch/standard library calls.
The goal is to make the optimization WORK, not to abandon it. Fix the bugs in the optimized code.

Use the SEARCH/REPLACE diff format to make targeted fixes:

<NAME>
fix_description
</NAME>

<DESCRIPTION>
Brief description of what you fixed. Explain how you preserved the optimization while fixing the issue.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# Fixed replacement code
>>>>>>> REPLACE

</DIFF>

Rules:
* PRESERVE the optimization approach - fix bugs, don't remove optimizations
* You may ONLY modify code that lies between "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" markers. Everything outside those markers is READ-ONLY.
* Do NOT include the markers themselves in your SEARCH/REPLACE blocks.
* The SEARCH section must match the original code EXACTLY (including whitespace/indentation)
* Make minimal, targeted changes - only fix what's broken
* You can include multiple SEARCH/REPLACE blocks for multiple fixes
* Focus on fixing compilation errors, syntax issues, and correctness problems
"""

REFINE_DIFF_USER_MSG = """# Code to Refine

The following code was generated and evaluated, but has issues that need fixing:

```{language}
{code_content}
```

# Evaluation Results

**Status**: {status}
**Performance Score**: {performance_score}

# Evaluation Feedback

{feedback}

# Task

Please provide SEARCH/REPLACE blocks to fix the issues identified above.
Make minimal, targeted changes to address the specific problems.

CRITICAL: 
- PRESERVE the optimization strategy - fix the bugs, do NOT revert to unoptimized code
- Only edit code within the EVOLVE-BLOCK-START/END regions
- Do not replace custom kernels/optimizations with vanilla implementations
"""


# =============================================================================
# FULL-BASED REFINEMENT (complete rewrite, more reliable but token-heavy)
# =============================================================================

REFINE_FULL_SYSTEM_MSG = """You are refining optimized code that was previously generated and evaluated.
Your task is to fix the issues identified in the evaluation feedback while PRESERVING the optimization strategy.

IMPORTANT: Do NOT revert optimizations or replace custom kernels with vanilla PyTorch/standard library calls.
The goal is to make the optimization WORK, not to abandon it. Fix the bugs in the optimized code.

Focus on:
1. Fixing compilation/syntax errors while keeping the optimization approach
2. Fixing correctness issues in the optimized implementation
3. Addressing specific issues mentioned in the feedback WITHOUT removing optimizations

Respond with the complete corrected code using this format:

<NAME>
refined_version
</NAME>

<DESCRIPTION>
Brief description of what you fixed. Explain how you preserved the optimization while fixing the issue.
</DESCRIPTION>

<CODE>
```{language}
# Your refined code here - must be complete and runnable
```
</CODE>

Rules:
* PRESERVE the optimization approach - fix bugs, don't remove optimizations
* You may ONLY modify code between "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" markers. Keep all code outside these markers UNCHANGED.
* Keep the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers in place
* Maintain the same inputs and outputs as the original program
* Make targeted fixes based on the feedback - don't rewrite everything unnecessarily
* Ensure the code compiles and runs correctly
"""

REFINE_FULL_USER_MSG = """# Code to Refine

The following code was generated and evaluated, but needs refinement:

```{language}
{code_content}
```

# Evaluation Results

**Status**: {status}
**Performance Score**: {performance_score}

# Evaluation Feedback

{feedback}

# Task

Please refine this code to fix the issues identified above.
Provide the complete corrected code that addresses the feedback.

CRITICAL:
- PRESERVE the optimization strategy - fix the bugs, do NOT revert to unoptimized code
- Only modify code within the EVOLVE-BLOCK-START/END regions
- Do not replace custom kernels/optimizations with vanilla implementations
"""


def format_refinement_status(results: dict) -> str:
    """Format the evaluation status for the refinement prompt."""
    correct_info = results.get("correct", {})
    is_correct = correct_info.get("correct", False)
    
    stdout_log = results.get("stdout_log", "")
    stderr_log = results.get("stderr_log", "")
    
    # Check for compilation errors
    has_compilation_error = False
    if stderr_log:
        error_indicators = ["error:", "Error:", "SyntaxError", "NameError", 
                          "TypeError", "ImportError", "ModuleNotFoundError",
                          "compilation failed", "CUDA error"]
        has_compilation_error = any(ind in stderr_log for ind in error_indicators)
    
    if has_compilation_error:
        return "COMPILATION ERROR"
    elif not is_correct:
        return "INCORRECT (validation failed)"
    else:
        return "CORRECT"


def format_refinement_feedback(results: dict) -> str:
    """Format evaluation feedback for the refinement prompt.
    
    Uses text_feedback from metrics which already contains formatted
    compilation errors, correctness errors, and profiling summaries
    from the evaluation script.
    """
    feedback_parts = []
    
    # Include text feedback - this already contains compilation errors,
    # correctness errors, and profiling summaries from evaluate.py
    metrics = results.get("metrics", {})
    text_feedback = metrics.get("text_feedback", "")
    if text_feedback and text_feedback.strip():
        feedback_parts.append(text_feedback.strip())
        feedback_parts.append("")
    
    # Include correctness details if not already in text_feedback
    correct_info = results.get("correct", {})
    if not correct_info.get("correct", False):
        feedback_parts.append("## Validation Status\n")
        feedback_parts.append("The program did not pass correctness validation.")
        if "message" in correct_info:
            feedback_parts.append(f"Message: {correct_info['message']}")
        feedback_parts.append("")
    
    if not feedback_parts:
        feedback_parts.append("No specific feedback available. Please review the code for potential issues.")
    
    return "\n".join(feedback_parts)


def build_refinement_prompt(
    code: str,
    results: dict,
    language: str = "python",
    mode: str = "diff",
) -> tuple[str, str]:
    """Build the system and user messages for refinement.
    
    Args:
        code: The current code to refine
        results: Evaluation results dict with correct, metrics, stdout_log, stderr_log
        language: Programming language
        mode: "diff" for token-efficient SEARCH/REPLACE, "full" for complete rewrite
        
    Returns:
        Tuple of (system_message, user_message)
    """
    metrics = results.get("metrics", {})
    combined_score = metrics.get("combined_score", 0.0)
    
    # Select prompt templates based on mode
    if mode == "diff":
        sys_template = REFINE_DIFF_SYSTEM_MSG
        user_template = REFINE_DIFF_USER_MSG
    else:  # "full"
        sys_template = REFINE_FULL_SYSTEM_MSG
        user_template = REFINE_FULL_USER_MSG
    
    sys_msg = sys_template.format(language=language)
    
    user_msg = user_template.format(
        language=language,
        code_content=code,
        status=format_refinement_status(results),
        performance_score=f"{combined_score:.2f}" if combined_score else "N/A",
        feedback=format_refinement_feedback(results),
    )
    
    return sys_msg, user_msg


# Backward compatibility aliases
REFINE_SYSTEM_MSG = REFINE_FULL_SYSTEM_MSG
REFINE_USER_MSG = REFINE_FULL_USER_MSG

