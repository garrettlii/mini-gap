"""Mini-GaP: laptop-scale Graph-as-Policy.

Layers (each only calls the one below it):
  sim -> skills -> graph -> agents -> rehearsal -> train / arbiter -> eval

Modules tagged OWNER: done are implemented infrastructure. Change them only with a reason
logged in docs/DECISIONS.md.
"""
