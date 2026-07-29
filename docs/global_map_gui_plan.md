# Global Map GUI Plan

Status snapshot: 2026-07-21 — first implementation complete

## Product decision

The map is a GUI view of the perspectives report, not a new pipeline module and not a map of every RSS source in the registry.

For each main story, it answers two different questions:

1. **Where is this story happening?** Event pins come from grounded `story_loci` produced by the existing perspectives planning call.
2. **Where did the retrieved reporting come from?** Selecting a story overlays the countries of the sources actually searched for that story.

Keeping these layers separate is essential. A story may happen in the Strait of Hormuz while useful reporting comes from the UAE, Qatar, the US, Pakistan, and elsewhere.

The source registry is currently considered healthy. Expanding it is not a prerequisite for this feature. Coverage shortcomings found through normal map use should drive later source additions.

## Current pipeline fit

The relevant pipeline flow is:

```text
briefs -> enrichment -> perspectives_report -> narrative_brief -> tts
```

`perspectives_report` is optional. When it runs, its JSON already contains the information needed for the selected-story reporting layer:

- stable `story_id`, title, and summary
- `selected_sources[]`, including country and language
- `source_yields[]`, including raw and retained counts
- `coverage_articles[]`, including source country and language
- `coverage_quality`, provider statuses, and framing synthesis

The missing information was an explicit distinction between a story's location and the countries selected for retrieval. Reusing `planner.target_tags` would be wrong because those tags describe where the pipeline wants reporting from, not where the event happened.

The existing perspectives planner now also returns:

```json
{
  "story_loci": [
    {
      "label": "Strait of Hormuz",
      "country": "",
      "kind": "event_site",
      "confidence": "high",
      "reason": "The reported shipping disruption occurs here."
    }
  ]
}
```

The planner is instructed to include only physical event sites or directly affected territory. Publisher origins, company headquarters, actor nationality, and countries that are merely interested are excluded. Non-geographic stories return an empty list. The model supplies semantic labels only; local GUI code owns coordinates.

`story_loci` is additive and appears both at the top level of each perspective story and in its public planner record. Older report JSON remains readable but has no event pins. Rerun the perspectives module for a date to generate the new field; the GUI does not silently infer locations from old headlines.

## Implemented architecture

The map remains a derived GUI read model:

```text
GET /api/map
GET /api/map?date=YYYY-MM-DD
```

`GuiDataService.map_snapshot(date)` selects a perspectives JSON file from the configured output directory and returns `gui_map.v1`. It does not write a new artifact or introduce another source of truth.

The response contains:

- available perspectives-report dates
- compact story title and summary data
- locally resolved medium/high-confidence event loci
- per-country coverage audit points
- compact article link metadata
- coverage quality, framing summary, search counts, and warnings

Large article bodies, context text, verification documents, and internal diagnostics are not sent to the map view.

Coordinates come from `mydailynews/gui/places.json`, which includes the countries used by the current source registry plus a deliberately small set of recurring waterways and regions. Resolution is exact:

1. exact named-place or alias match
2. supplied country code to country centroid
3. unresolved, with a visible warning

There is no live geocoder, fuzzy matching, GIS database, frontend framework, map-tile request, or mapping dependency.

## Coverage audit semantics

For the selected story, source countries use three user-facing states:

| Display state | Meaning |
| --- | --- |
| Coverage found | At least one retained coverage article came from the country. |
| Searched, none retained | One or more sources in the country were selected, but no article survived retrieval and filtering. |
| Unmarked / not searched | The perspectives planner did not select a source in that country. This does not mean the country had no reporting. |

This is the most important qualification in the design. The perspectives planner decides which countries and regions to search, so the map visualizes the pipeline's observed footprint, not exhaustive global media coverage. The details panel states this explicitly.

Provider failures remain visible at story level, but v1 does not assign a provider-wide failure to a particular country when the artifact cannot support that attribution.

## Implemented GUI

Map is a top-level GUI view alongside Reports and the existing administrative views.

Desktop layout:

```text
story list | offline SVG world map | selected-story details
```

The view provides:

- newest perspectives date by default and a date selector
- all resolved story pins for the date
- click-to-select from either the map or story list
- a toggle for selected-story coverage points
- distinct markers for event, coverage found, and searched-empty
- source-selection and country-result counts
- found and searched-empty country lists
- framing synthesis and compact coverage article links
- unresolved stories retained in the list with an explanation
- a stacked narrow-screen layout

The world outline and equirectangular projection are embedded in the local JavaScript:

```text
x = ((longitude + 180) / 360) * width
y = ((90 - latitude) / 180) * height
```

The outline is intentionally low detail. Its purpose is navigation and comparison, not precise cartography.

## Files changed

```text
mydailynews/pipeline/perspectives_report.py  # story_loci schema, normalization, and output
mydailynews/ai/prompts.py                    # grounded story-location instruction
mydailynews/gui/data.py                      # gui_map.v1 read model
mydailynews/gui/server.py                    # GET /api/map route
mydailynews/gui/places.json                  # local country/place coordinates
mydailynews/gui/static/index.html            # Map view markup
mydailynews/gui/static/gui.css               # map layout and markers
mydailynews/gui/static/js/state.js           # map state
mydailynews/gui/static/js/main.js            # lazy view loading and refresh
mydailynews/gui/static/js/map.js              # SVG rendering and interactions
tests/test_gui_data.py                       # map contract and coverage-state test
tests/test_perspectives_report_module.py     # story_loci planner/output checks
```

## Deliberate first-version limits

- No deterministic location baseline was added.
- No new source packs or source-registry expansion.
- No persisted `global_map` artifact or pipeline module.
- No inferred event pins for older perspectives reports.
- No country polygons; coverage is shown at country centroids.
- No pan/zoom, clustering, timeline, or animated connection lines.
- No framing colors inferred from prose.
- No claim that unsearched countries lack coverage.

## What to evaluate before expanding it

Use normal perspectives runs to evaluate the product rather than adding infrastructure first:

- How many important stories receive a useful medium/high-confidence locus?
- Are recurring cities, waterways, or regions missing from the local lookup?
- Does the searched-empty state help explain gaps, or is source-level detail needed?
- Does the planner repeatedly overlook countries whose perspectives the user expects?
- At actual daily story volume, do pins overlap enough to justify clustering or zoom?

If planner country selection is the dominant weakness, improve retrieval planning with broader or rule-assisted country selection while preserving the `not searched` distinction. If location resolution is the dominant weakness, expand the small local place lookup first. Add a separate location call only if those simpler changes prove insufficient.
