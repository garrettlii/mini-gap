"""GRPO reward: validate + rehearse the edited graph.

Registration pattern for mlx-lm-lora (verify names against your installed version
before relying on them; the library changes quickly):

    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function

    @register_reward_function()
    def sim_edit_reward(prompts, completions, answer, types=None):
        # batched lists in, one float per completion out
        ...

The reward only sees prompts, completions, and the dataset's answer field. Store a
lookup key in `answer` (for example "place/iter07/graph_ab12"), and load the saved
graph + rehearsal seeds from disk with it.

Reward table (from the build guide):
    not parseable JSON                  0.0
    parses but fails validation         0.05
    valid                               0.1 + (success_after - success_before) on 8 seeds
    valid but cycle time grows > 50%    subtract 0.2

Cache by (graph hash, edit hash): GRPO samples the same edit often.
Read 20 sampled edits by hand after every run to catch reward hacking
(deleting verify_grasp, maxing retries, inflating wait_steps).
"""

from __future__ import annotations


def score_edit(answer_key: str, completion: str, n_seeds: int = 8) -> float:
    """Pure function the registered reward wraps. Unit-test this directly."""
    raise NotImplementedError
