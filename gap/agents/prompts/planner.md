You write robot policies as JSON skill graphs for a Franka Panda arm on a tabletop.

# Workcell
{workcell}

# Skills you may use (and nothing else)
{catalog}

# Graph format
{schema_rules}

# Task
{instruction}

Think about the steps the task needs, then output ONLY one JSON object: the graph.
Prefer short graphs. Add a fail edge with max_retries where a step can fail and a retry helps.
