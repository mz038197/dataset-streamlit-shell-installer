# Issue tracker: GitHub

Issues and specs (you may know a spec as a PRD) for this repo live as GitHub issues in `mz038197/dataset-streamlit-shell-installer`. Use the `gh` CLI for every operation.

## Conventions

- One feature = one **parent issue** whose body is the spec. Implementation tickets are **sub-issues** of that parent, one ticket per issue — never a single combined tickets issue.
- Create: `gh issue create --title "..." --body-file <file> --label <triage-label>`
- Link a sub-issue (needs the child's numeric id, not its number):
  `gh api repos/{owner}/{repo}/issues/<parent-number>/sub_issues -F sub_issue_id=$(gh api repos/{owner}/{repo}/issues/<child-number> --jq .id)`
- Triage state is a label (see `triage-labels.md` for the strings). Swap with `gh issue edit <n> --add-label ... --remove-label ...`.
- Blocking: a `Blocked by: #NN, #NN` line near the top of the body. A ticket is unblocked when every issue it lists is closed.
- Comments and conversation history are issue comments (`gh issue comment <n> --body-file <file>`).
- Write issues in Traditional Chinese, matching `CONTEXT.md`. Link ADRs by repo path (`docs/adr/NNNN-....md`).

## When a skill says "publish to the issue tracker"

Create a GitHub issue (and, for a multi-ticket feature, a parent issue plus sub-issues).

## When a skill says "fetch the relevant ticket"

`gh issue view <n> --comments`. The user will normally pass the issue number or URL directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a parent issue with one **child** sub-issue per ticket.

- **Map**: a parent issue whose body holds the Notes / Decisions-so-far / Fog sections.
- **Child ticket**: a sub-issue of the map with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`).
- **Blocking**: a `Blocked by: #NN, #NN` line near the top of the body.
- **Frontier**: open, unblocked, unassigned sub-issues of the map; lowest number wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` before any work.
- **Resolve**: post the answer as a comment under an `## Answer` heading, close the issue, then append a context pointer (gist + link) to the map's Decisions-so-far by editing the parent body.
