# Code notes

The code is intentionally kept simple and project-specific. Most comments explain why a step exists, not what every Python line does.

A few practical design choices:

- memory banks are compact so the project fits on a 4 GB GPU,
- category detection uses prototypes instead of training another classifier,
- category match scores are shown as ranking scores, not calibrated probabilities,
- notebooks keep the experimental trail, while `src/`, `scripts/`, and `app/` contain reusable code.
