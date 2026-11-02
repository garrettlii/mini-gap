You improve the structure of a robot skill graph from a failure report.

# Current graph
{graph_json}

# Skills available
{catalog}

# Failure report across rehearsal instances
{report}

If failures come from a missing step, wrong ordering, or no recovery path, propose up to
{max_edits} structural edits ("insert_after", "add_edge", "remove_node"), each with a
one-sentence "rationale". If parameter tuning would fix it instead, return {"edits": []}.
Output only {"edits": [...]}.
