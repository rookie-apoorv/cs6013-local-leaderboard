# CS6013 local leaderboard (public view)

A public, daily-updated view of a **local** evaluation of compressed Qwen3.5-4B models
(CS6013 course project: 40 % / 20 % / 10 % of the original weights). It is not the
official course leaderboard.

Every model is run with the course grader's settings and scored with the grader's own
answer parser; the page reports avg@k with 95 % intervals, paired differences against a
reference model, and behaviour statistics (cut-off rate, tokens, rambling). See the
page's FAQ for details.

## How it is built

`lb_sync.py` reads the evaluation registries from a private Hugging Face dataset (scores and
token counts only — no model outputs), removes identifying details (models appear under
short aliases; no repository ids, problem texts or answers) and writes a static site
(`index.html` + `lb_data.js`). The workflow in `.github/workflows/publish.yml` does this
once a day and on demand, and deploys it with GitHub Pages.

## One-time setup (repository owner)

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. **Settings → Secrets and variables → Actions:**
   - *Secret* `HF_TOKEN` — a Hugging Face **read** token that can read the private dataset.
   - *Variable* `LB_REPO` — the dataset id, e.g. `<hf-user>/cs6013-local-leaderboard`.
3. **Actions → Publish leaderboard → Run workflow.** The site appears at
   `https://<github-user>.github.io/<repo>/` after about a minute.

## Local use (owner)

```
pip install -r requirements.txt
python lb_sync.py --repo <hf-user>/cs6013-local-leaderboard          # private view
python lb_sync.py --serve                                            # + trace viewer, saved sets
python lb_sync.py --export site --public                             # preview the public site
```

Question sources: AIME 2022–2026, HMMT, BRUMO, CMIMC, SMT (via MathArena, CC BY-NC-SA 4.0)
and MATH-500, referenced by id only.
