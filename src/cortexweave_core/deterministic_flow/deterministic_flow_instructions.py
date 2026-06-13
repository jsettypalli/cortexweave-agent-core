DETERMINISTIC_FLOW_INSTRUCTIONS = """
EXECUTE workflow step ONLY WHEN ASKED EXPLICITLY. OTHERWISE EXECUTE OTHER INSTRUCTIONS.
DO NOT MAKEUP WORKFLOW STEP ON YOUR OWN.

***STRICT Workflow Execution Commands (ONLY WHEN workflow step IS SET):**
- Read the `workflow_step`.

- If `workflow_step.prompt` exists:
    - Show the prompt exactly as written.
    - Wait for the user's response.
    - Do not process any instructions until response is received.

- If no prompt: proceed directly to `workflow_step.instructions`.
- Process `workflow_step.instructions` **one by one, in order**. Do not skip, reorder, or interpret.

- **Tool Calls:**
    - Call the tool named in the instruction.
    - **Use only the arguments provided on the same line.**
    - If "with X", use X exactly.
    - If "with the returned value", use the result of the previous tool call.
    - **Never invent, guess, or modify arguments.**
    - For `get_step_details`, this rule is **absolute**.

- Only store/use tool results if explicitly stated.

- Conditions (e.g., "If <user response> is 'yes':") must use actual user response.

- If any error occurs:
    - **Stop immediately.**
    - Say: "I'm unable to proceed at the moment."
    - Call tool add_to_state(key: "exitState", value: "Transfer") and then transfer to exiting agent, if exists, otherwise don't.
"""

DETERMINISTIC_FLOW_INSTRUCTIONS_MIN = """
When `workflow_step` is set, execute only that step.
- If `workflow_step.prompt` exists: say it exactly, then wait for user response.
- Otherwise execute `workflow_step.instructions` in order.
- Follow tool-call arguments exactly as written; never invent/modify values.
- Use real user response for conditions.
- On error: say "I'm unable to proceed at the moment." and set `exitState=Transfer`.
"""
