You repair one node of a robot skill graph.

# The node
{node_json}

# Its skill
{skill_entry}

# What happened when this node ran (across rehearsal instances)
{node_report}

Propose exactly ONE edit to this node, as a JSON object with "op" set to "set_param" or
"swap_skill", plus a one-sentence "rationale". Stay inside each param's declared range.
Output only the JSON object, in one of these two forms:
{"op": "set_param", "node": "{node_id}", "param": "<param name>", "value": <number>, "rationale": "<why>"}
{"op": "swap_skill", "node": "{node_id}", "skill": "<skill name>", "rationale": "<why>"}
