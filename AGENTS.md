# Project Codex model routing

- Use `gpt-6-luna` for non-Spark child agents and the Luna fallback in this repository. This overrides older `gpt-5.6-luna` references in user-level guidance.
- Keep Spark and explicitly selected models as configured. The local `.codex/config.toml` sets matching runtime defaults.
