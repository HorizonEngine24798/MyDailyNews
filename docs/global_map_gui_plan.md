# Global Map GUI Plan

The global map should become a central way to browse MyDailyNews, not just a decoration beside reports.

The useful split is:

- **News Map**: where key stories are happening or focused.
- **Coverage Map**: where coverage is coming from, and how framing differs.

The map should work from local report JSON. Do not require live map tiles, geocoding APIs, or a new database in v1.

## Current State

The GUI already has:

- A reports view.
- A report list and Markdown viewer.
- A `perspectives_report` run kind.
- JSON artifacts for briefs, enrichment, narrative brief, and perspectives report.

Missing:

- No map-first report browser.
- No story location data.
- No source-country coverage map.
- No shared map data contract.
- No country/city centroid lookup.

## Product Goal

The map should answer two different questions without mixing them up:

1. **What is happening where?**
2. **Who is covering it from where?**

For a story about the Red Sea, the event locus may be the Red Sea, Yemen, Egypt, shipping lanes, and Gulf ports. Coverage origins may be GB, US, AE, IN, QA, FR, CN, and others.

That distinction is the feature.

## Main Concepts

### Story Loci

Places the story is about.

```json
{
  "label": "Red Sea",
  "country": "",
  "lat": 20.0,
  "lon": 38.0,
  "confidence": "medium",
  "kind": "event_area",
  "reason": "shipping disruption location"
}
```

Rules:

- A story can have multiple loci.
- Low-confidence loci should not be plotted by default.
- Keep the `reason`; it makes the map auditable.
- Use country/city centroids first, not live geocoding.

### Coverage Points

Places where reporting sources are based.

```json
{
  "country": "IN",
  "lat": 20.5937,
  "lon": 78.9629,
  "article_count": 5,
  "sources": ["The Hindu", "Indian Express"],
  "languages": ["English", "Hindi"],
  "framing_summary": "Focuses on route costs and export delays."
}
```

Rules:

- Coverage origin is not the same as story location.
- Use source country from the perspectives coverage artifact when available.
- If source country is unknown, keep the article in the list but do not map it.

### Story Links

Lines between story loci and coverage origins.

Use lines only when a story is selected. Showing every line for every story will become visual soup.

## Data Contract

Add a map artifact:

```text
output/YYYY-MM-DD_global_map.json
```

This should be a derived artifact, not a new source of truth. It can be generated from enrichment, narrative brief, and perspectives/global coverage artifacts.

Shape:

```json
{
  "schema": "global_map.v1",
  "date": "2026-07-09",
  "generated_at": "2026-07-09T00:00:00Z",
  "stories": [
    {
      "story_id": "story-1",
      "title": "Red Sea shipping disruption",
      "summary": "",
      "source_report_ids": ["2026-07-09_perspectives_report.json"],
      "loci": [],
      "coverage_points": [],
      "coverage_articles": [],
      "framing": {},
      "warnings": []
    }
  ],
  "warnings": []
}
```

Why a separate map artifact:

- The GUI gets one predictable file to load.
- The main news map can work even when perspectives/global coverage is disabled.
- The coverage/framing layer can enrich the same story later.
- Report formats can evolve without forcing the map UI to know every report shape.

## Source Diversity

The map only becomes useful if source coverage is globally diverse.

Use this order:

1. **Existing selected stories** from briefs/enrichment.
2. **Perspectives coverage** per key story.
3. **Curated RSS packs** by region/language.
4. Paid/event APIs only if the provider path is not enough.

### Perspectives Coverage

Use the perspectives report coverage records for coverage origins and global article discovery:

- source country
- source language
- article list
- recency window

Use the curated source registry as the maintained source-origin catalog.

### Regional RSS Packs

Add optional curated packs later:

```text
config/sources/world_english.json
config/sources/middle_east.json
config/sources/europe.json
config/sources/asia.json
config/sources/africa.json
config/sources/latin_america.json
```

Each source should include:

- name
- URL
- region
- country
- language
- category
- enabled

Do not import a giant feed list first. Start with 5-10 reliable sources per region.

## Location Extraction

Start with the enrichment story threads.

For each key story, extract loci from:

- story title
- selected article titles/snippets
- enrichment confirmed facts
- narrative brief section text if available

Implementation options:

1. Heuristic country/city lookup from known names in text.
2. LLM JSON extraction with strict schema.
3. Both, with the LLM allowed to choose from candidates.

Lazy v1:

- Add a small local `country_centroids.json`.
- Add a short city/region list only for common geopolitical locations.
- Ask the LLM for `label`, `country`, `kind`, `confidence`, and `reason`.
- Resolve only labels found in the local lookup.
- Drop unresolved or low-confidence loci.

No live geocoder in v1.

## GUI Layout

Make the map its own top-level tab:

```text
Reports | Map | Memory | Run
```

The map tab should have:

- left story list
- central world map
- right details panel
- compact layer toggle

Layer toggles:

- `Stories`: story loci
- `Coverage`: coverage origins for selected story
- `Framing`: coverage origins colored by framing cluster
- `Gaps`: expected-but-thin regions
- `Sources`: raw article/source list

Default:

- show `Stories`
- select the top story
- show its details in the side panel

When a story is selected:

- highlight its locus/loci
- show coverage origin points if available
- draw lines from coverage origins to story loci
- list source articles under the map or in the side panel

## Visual Design

Use a static, dependency-free map first:

- equirectangular projection
- local SVG or CSS-backed world outline
- country centroid dots
- SVG lines
- no map tiles
- no network calls in the GUI

This is enough to test the product idea.

Only add Leaflet or another map library if:

- dot overlap is bad
- panning/zooming becomes necessary
- country-level centroids feel too crude

## Interaction Details

Story dot:

- hover: title, one-line summary, locus label
- click: select story

Coverage dot:

- hover: country, article count, languages
- click: source list and framing note

Line:

- show only for selected story
- thin and low-contrast by default
- highlight on hover

Story list:

- sortable by importance, region, source count, freshness
- filter by region/language/source country

Details panel:

- story summary
- loci
- coverage counts
- framing summary
- article links
- warnings

## Bias/Framing Integration

The perspectives/global framing module should produce coverage data. The map GUI should display it.

`perspectives_report` should not own the map UI. It should write:

- `coverage_points`
- `coverage_articles`
- `framing`
- `coverage_counts`

The global map artifact should merge those into `global_map.v1`.

This keeps the GUI usable for normal news even when `perspectives_report` is disabled.

## Implementation Phases

### Phase 1: Static News Map

- Add `docs/global_map_gui_plan.md`.
- Add `output/YYYY-MM-DD_global_map.json` generation from enrichment stories.
- Add `story_loci` extraction using a tiny local lookup.
- Add a `Map` tab.
- Render story points only.

Success: the main daily stories appear on a world map.

### Phase 2: Coverage Overlay

- Add coverage points from expanded `perspectives_report`.
- Add story selection.
- Draw source-country points for selected story.
- Draw simple lines from coverage origins to story loci.

Success: selecting a story shows where coverage is coming from.

### Phase 3: Source Diversity

- Add selected RSS coverage fallback.
- Add source-country and language counts.
- Add "thin coverage" warnings.
- Add optional curated regional RSS packs if the seed registry misses obvious sources.

Success: the map is not just U.S./UK dots.

### Phase 4: Framing Layer

- Add framing summaries by source country/language.
- Color coverage dots by framing group or emphasis.
- Show framing notes in the side panel.

Success: the map shows not only who covered the story, but how coverage differed.

### Phase 5: Polish

- Add filters.
- Add timeline scrubber if daily maps accumulate.
- Add better local gazetteer entries.
- Consider Leaflet only if the static map becomes limiting.

## Non-Goals

- No live geocoding API in v1.
- No live map tiles in v1.
- No full GIS stack.
- No exact event geolocation for ambiguous stories.
- No aggressive clustering.
- No maintained global source-origin catalog.
- No map dependency before the static map proves insufficient.

## First Useful Map

The first useful map is successful when:

- today's key stories appear as map points
- clicking a point opens the story summary and articles
- a global/framing report can add source-country coverage points
- coverage origins and story loci are visually distinct
- no network access is needed to view the map

Shortest next step: create `global_map.v1` from enrichment stories and render story loci in a new GUI `Map` tab.
