# Personal memory bundle

These files are a snapshot of Claude Code's per-project memory from the old machine, as of 2026-05-28.

**They do NOT travel with the repo by design** — they live in `~/.claude/projects/<project-key>/memory/` on your workstation. I'm bundling them here only so you can transfer them to the new machine, then delete this folder.

## How to restore on the new machine

After `git clone`-ing this repo on the new machine, copy these files into your local Claude Code project memory directory:

**On Mac/Linux:**
```bash
# After cloning the repo on the new machine
PROJECT_KEY=$(pwd | sed 's|/|-|g')   # rough mapping; the actual key uses the absolute path
mkdir -p ~/.claude/projects/"$PROJECT_KEY"/memory/
cp docs/handoffs/personal-memory/*.md ~/.claude/projects/"$PROJECT_KEY"/memory/
# Then delete the bundle so it doesn't drift from the live memory:
git rm -r docs/handoffs/personal-memory/
git commit -m "chore: remove personal-memory bundle after transfer"
```

**On Windows (PowerShell):**
```powershell
$ProjectKey = (Get-Location).Path -replace '[:\\/]', '-'
$Dest = "$HOME\.claude\projects\$ProjectKey\memory"
New-Item -ItemType Directory -Force $Dest | Out-Null
Copy-Item docs\handoffs\personal-memory\*.md $Dest
# Then:
git rm -r docs/handoffs/personal-memory/
git commit -m "chore: remove personal-memory bundle after transfer"
```

The exact project-key folder name is whatever Claude Code generates from the absolute path (it's usually `<full-path-with-slashes-as-dashes>`). The easiest way to find it on the new machine is to start a Claude Code session in the repo first, then check what folder appeared under `~/.claude/projects/`.

## What's here

| File | Type | What it tracks |
|------|------|----------------|
| `MEMORY.md` | index | One-line pointers to every memory file (loaded into every conversation) |
| `project_batches_complete.md` | project | Batches A–E shipped (+8.7pp WF accuracy vs baseline) |
| `project_bot_data_sync.md` | project | Bot only syncs on /predecir_progol; other commands hit local DB |
| `project_concurso_2333_plan.md` | project | The 4-day plan we executed this week (Mon 25 – Thu 28) |
| `project_progol_bot_vm.md` | project | Bot VM provisioning notes (slim deps, manage_users CLI) |
| `project_progol_trainer_vm.md` | project | Trainer VM is STANDARD (Spot truncated runs) |
| `reference_gcs_bucket.md` | reference | `gs://progol-data-storage` prefixes + dual-auth quirk |
| `reference_workstation_ssl.md` | reference | Windows SSL workarounds (certifi, http.sslVerify=false) |

## After transferring

Delete this folder once you've copied the files. Keeping a stale snapshot in git will drift away from the live memory and create confusion. The handoff doc (`docs/handoffs/2026-05-28-handoff.md`) is the persistent record; this folder is one-time transit.
