# CLI

`main.py` runs either the configured pipeline series or one standalone module.

## Normal Run

```bash
python main.py --config config.local.json
```

By default this runs `pipeline.default_series` from the config.

## Run One Module

```bash
python main.py --config config.local.json --module briefs
python main.py --config config.local.json --module enrichment --date YYYY-MM-DD
python main.py --config config.local.json --module narrative_brief --date YYYY-MM-DD
python main.py --config config.local.json --module perspectives_report --date YYYY-MM-DD
python main.py --config config.local.json --module tts --markdown-path output/YYYY-MM-DD_general_brief.md
```

Modules are `briefs`, `enrichment`, `narrative_brief`, `perspectives_report`, and `tts`.

`--date` is only for standalone `enrichment`, `narrative_brief`, `perspectives_report`, and `tts`; omit it to use today. Standalone modules reuse same-day artifacts from `output/` when needed.

## Common Options

```bash
python main.py --config config.local.json --module series --skip-module tts
python main.py --config config.local.json --brief general
python main.py --config config.local.json --debug
python main.py --config config.local.json --list-stages
```

Memory commands:

```bash
python main.py --config config.local.json --memory inspect
python main.py --config config.local.json --memory prune
python main.py --config config.local.json --memory export --memory-export memory.json
```
