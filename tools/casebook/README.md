# The casebook design system

What every chapter of the Corporate Default Prediction casebook looks like, and the code that builds it.

The reference is [Chapter 1: The Panel](../../notebooks/01_the_panel.ipynb). When a rule here and that notebook disagree, the notebook is wrong: fix its source and rebuild.

> This casebook is educational content created in a personal capacity

> Nothing here is investment advice or a recommendation to buy or sell any security

> The data is a public research dataset with anonymised company names

## The one rule

**Words are typed. Structure is drawn.**

Everything a reader takes in as a sentence is plain markdown: headings, questions, readings, tables, findings. Everything that gives the page its identity and its bearings is a picture drawn by this folder: the cover, the route, the part openers, the pitfall plates, the diagrams, the card to the next chapter.

A notebook is read in VS Code, in JupyterLab, and on GitHub, in a light theme and a dark one. An embedded picture and plain markdown are the only two things all of them show the same way. So the design never depends on CSS inside a cell, and a reader who opens a text cell sees words, not markup.

A chart is neither typed nor drawn here. It is the real output of a code cell, in the house style the chapter itself sets.

## Identity

| Token | Value | Used for |
| --- | --- | --- |
| Carbon | `#121316` | Covers, the hero, the next card |
| Carbon rule | `#2E3138` | Hairlines on carbon, and the edge of every carbon component |
| On carbon | `#F2F0EA`, soft `#B9BCC4`, mute `#7E838D` | Text on carbon |
| Sheet | `#F7F7F5` | Every exhibit, plate, and diagram |
| Sheet deep | `#ECECE8` | Footers, table bands |
| Rule | `#CFCFC9` | Hairlines on sheet |
| Ink | `#16181D`, slate `#4F5560`, mist `#8B909A` | Text on sheet |
| Signal | `#D9482B`, bright `#FF6A4D` on carbon | Failure, a pitfall, and the house mark |
| Steel | `#295F92`, bright `#8FB1D1` on carbon | Survival, and place ("you are here") |

Charts use two palettes that never share a chart. Outcome charts: signal for failed, steel for survived, `#878C96` for censored or any other exit. Model charts: classic scores `#878C96`, logistic `#295F92`, trees `#B48219`, networks `#8C6FB9`, blend `#16181D`.

| Typeface | Role |
| --- | --- |
| Archivo, weights 600 and 800 | Titles, chapter numerals, the numbers on a plate |
| Public Sans, weights 400 and 600 | Labels, captions, all chart text |
| JetBrains Mono, weights 500 and 700 | Numbers, tags, tick labels, times |

All three are under the SIL Open Font License and live in `assets/fonts` as static files with their licences. The same files serve the drawn components and the charts. Corners are square and nothing casts a shadow: hairlines and flat colour only.

**The house mark.** One step curve is the mark of the casebook. It is the share of this dataset's companies not yet failed: one minus the cumulative incidence of failure, with every other exit as a competing risk, on an axis that is truncated and labelled at both ends. `components.house_mark` computes it from the event table and `components.mark_curve` draws it. It sits on every cover with a marker at the chapter's place, first chapter at the left and last at the right. It is never typed and never redrawn by hand.

## Anatomy of a chapter

A chapter is built in this order. "Drawn" parts come from `components.py` by way of `apply.py`. "Typed" parts are markdown in the chapter source.

| Part | Form | Rule |
| --- | --- | --- |
| Cover | Drawn: `cover` | First cell. The numeral, the title, the chapter's question, the house mark with this chapter marked, the author, and the minutes. |
| Title cell | Typed | `# Chapter N: Title`, a one-line purpose, then the disclaimer as three quoted lines: one sentence to each, a blank line between, no closing full stop. |
| The Question | Typed | `## The Question`: the situation in a few short lines. `## What You Will Learn`: three to five objectives. `## What You Need`: the chapters and artifacts this one reads. |
| Route | Typed heading, drawn `route` | Under `## Your Route`: the parts in order on a rail, the minutes under each, then one line of time. |
| Part opener | Drawn: `opener`, in a cell of its own | Before each `# ` part heading: which part of how many, its goal in one short sentence, about how many minutes. |
| Part | Typed | At most 12. The first is `# Setting Up`. A part heading starts its cell. |
| Setting Up | Typed, code | Three things, in this order. The kernel line: select the Jupyter kernel `corp-default`, in those words ("Jupyter kernel", not only "kernel"), then a link to the README's Quick Start for trouble. The data dictionary, under `## The Data Dictionary`: a typed table of every column the chapter loads that the reader has not met, before any code touches it. Then the load and a plain first look: the first rows with `head()`, the size and column types with `shape` and `info()`, a summary with `describe().T`. No slice or trick comes before the reader has seen the whole table. |
| Step | Typed, code, typed | A line starting `Q. `, a code cell, then `**Reading.**` and one or two lines on what the output shows. A bold sentence hands off to the next step. |
| Predict | Typed | `**Predict N.**` asks the reader to commit to a number before a key result. At least two a chapter. |
| Exhibit | A code cell's output | A chart framed by `viz.exhibit`: an EXHIBIT n tag, a title that states the finding, with its number when the finding is a number (computed in the title's f-string, never typed), one line on what to notice, a source line. Numbered from 1 in each chapter. The text refers to it as "Exhibit 4". |
| Pitfall | Drawn: `pitfall` | Where a naive analysis gives a different number: the naive number beside the honest one, each labelled, with one line on why. Both are computed in the chapter. At least one a chapter from Chapter 2 on. |
| Diagram | Drawn: a function in `chapters/diagrams.py` | Boxes, words, arrows, and time bars on a plate with its caption. |
| Check | Typed | `**Check.**`, one question, and the answer folded under "Show the answer". At least two a chapter. |
| Findings | Typed | `# Findings`: a numbered list, each item citing its exhibit. |
| Your Turn | Typed | `# Your Turn`: three extensions, each with its hint folded under "Hint". |
| Next | Drawn: `next_card` | The last cell: the card to the next chapter, as a link. Chapter 7 links to the landing page. |

Prose rules: a paragraph is one or two sentences, then a blank line, because a reader of a notebook does not follow a long run of sentences; one sentence to a line in the source; bullets only for parallel items under a lead-in; a term is bold where it is first named; no em dashes or en dashes; no emojis; no version history; no number that is not a placeholder.

## Numbers

No number is typed in a chapter's prose. A code cell calls `store.put("key", value)`, and the prose writes `{{chN.key}}` or `{{chN.key:.1%}}`, with any Python format after the colon. The build fills every placeholder after the chapter has run and stops on a key no cell stored. A plan's words are filled the same way, so the two numbers on a pitfall plate are computed too.

## The drawn components

Each function in `components.py` returns the HTML of one component, styled by `design.css`. `Studio.png` draws it at twice its display size, checks that every word fits, and saves a PNG. Every class name carries the `cb-` prefix.

| Function | What it draws |
| --- | --- |
| `house_mark(events)`, `mark_curve(mark, chapter)` | The house mark: its points from the event table, then the curve with a chapter marked. |
| `step_curve(points, width, height, box, ...)` | Any step curve, as an SVG. `mark_curve` is built on it. |
| `cover(number, title, question, minutes, curve)` | The cover. |
| `route(parts, total)` | The parts on a rail, labels alternating above and below. Up to 12 parts. |
| `opener(index, count, goal, minutes)` | The strip before a part. A goal is at most eight words. |
| `pitfall(name, naive_label, naive_value, honest_label, honest_value, why)` | A pitfall plate. |
| `stage`, `box`, `words`, `arrows`, `timebar` | A diagram, placed by position. `timebar` kinds: `train`, `embargo`, `valid`, `holdout`, `label`. |
| `figure_plate(body, caption)` | The sheet frame a diagram sits in. |
| `next_card(label, plate, title, line)` | The card to the next chapter. |

## Exhibits

`viz.use_house_style()` sets the matplotlib style, registers the fonts, and starts exhibit numbering at 1. `viz.exhibit(fig, title, note, source, save_as)` frames a finished chart. Give a figure `figsize=(viz.EXHIBIT_WIDTH_IN, height)`: 8 inches at 150 dpi is 1200 px.

Chart rules: one message to an exhibit; direct labels in place of a legend where they fit; a signed quantity on a symmetric-log axis (`viz.symlog_axis`); a log axis set with `viz.log_axis`, so its ticks read as numbers a reader can say (10,000 and 0.01), never as powers of ten; an interval wherever a rate or a metric is shown; no dual axes; no default colour cycle, so every colour is a choice.

Jupyter crops a figure to its ink by default, which would give every exhibit a different width. `use_house_style` replaces that with the whole canvas, and gives each exhibit its alt text: `Exhibit n: title. note`. The tests, the time rule, and `check --text` all read an exhibit by that alt text.

## The time rule

Times on a cover, a route, and an opener come from one rule, so they mean the same thing in every chapter: 150 words a minute, half a minute for each code cell, one minute for each exhibit, three minutes for each extension in Your Turn. A part's time is a whole number of minutes, at least one. The cover's total is rounded to five. A chapter takes at most 60 minutes by this rule, Your Turn included: a test reads the time from the cover and fails a longer chapter. A picture is read by its alt text; picture data and markup are not counted.

## Building a chapter

```
uv run python -m tools.casebook.build 03          # one chapter
uv run python -m tools.casebook.build --all       # every chapter that has a source, in order
uv run python -m tools.casebook.check 03 --sheet artifacts/sheets/03
uv run python -m tools.casebook.check 03 --text
```

| File | What it does |
| --- | --- |
| `build.py` | Reads `chapters/NN_name.py`, executes it from the repo root, fills its placeholders, applies the design, and writes `notebooks/NN_name.ipynb`. Prints one line: parts, exhibits, minutes, megabytes, seconds. |
| `apply.py` | Puts an executed notebook into the design from its plan. Reads the parts, their length, and the exhibits from the notebook itself. |
| `check.py` | Lists a built chapter's parts with their goals and minutes, runs its tests, and saves every picture and exhibit, in order, on sheets 1200 px wide. `--text` prints the notebook as plain text, each picture as its alt text. |
| `studio.py` | The headless browser that draws a component and refuses words that do not fit. |
| `proof.py` | The proof sheet of the identity review. |

The build is the only path to a notebook. A notebook is never edited by hand, and a rebuild of an unchanged chapter writes the same file.

A chapter source is a jupytext percent file. Its first code cell calls `store.open_chapter(N)` and `viz.use_house_style()`. Two markers, each alone in a markdown cell, stand where a drawn plate goes:

```
<!-- pitfall: key -->
<!-- diagram: name -->
```

A plan, `chapters/plans/NN.json`, holds only what a person writes:

| Key | What it holds |
| --- | --- |
| `number`, `title`, `question` | The cover. The title cell of the source must read `# Chapter N: Title`. |
| `goals` | One goal for each part, in order: one sentence of eight words or fewer. |
| `pitfalls` | For each `<!-- pitfall: key -->`: `name`, `naive_label`, `naive_value`, `honest_label`, `honest_value`, `why`. Labels are typed in sentence case and drawn in capitals. Values are placeholders. |
| `next` | `title` and `line` of the next card. The link is the next chapter's notebook, named from its number and title. |

A diagram needs no plan entry: `<!-- diagram: name -->` calls the function `name` in `chapters/diagrams.py`, and its caption is its alt text.

When the plan and the notebook disagree, the build stops with one sentence that says what to change: a goal count that does not match the parts, a pitfall marked and not described, a title that differs, a key the plan does not take.

## Rules that keep it working everywhere

1. A drawn component is embedded as a markdown image, `![alt](data:image/png;base64,...)`. No `<img>` tag and no `style` attribute in a cell, ever.
2. Every drawn component is 1200 px wide and drawn at 2400 px. Every exhibit is 1200 px wide. So every picture spans the column.
3. Every picture has an opaque background. Nothing relies on the colour of the page behind it, and a carbon component carries a 1 px carbon-rule edge so it keeps its outline on a dark page.
4. Every picture has alt text that says what the picture says. An opener's is its part, goal, and minutes. A pitfall's carries both numbers. An exhibit's is its number, title, and note.
5. A component's height is a whole number of pixels. `Studio.png` refuses anything else, because a part pixel shows as a stray line.
6. Text inside a picture is short: a label, a caption, a goal. A sentence that teaches is typed. A component refuses words that do not fit: shorten the words, never shrink the type.
7. Numbers on a picture are computed, never typed.
8. Colour carries one meaning each: signal is failure, steel is survival and place, and the model palette never appears on an outcome chart.

Install the author tools once: `uv sync`, then `uv run python -m playwright install chromium`.

## The landing page

`uv run python -m tools.casebook.landing` draws the pictures of the repo's `README.md` into
`assets/readme/landing/`, 830 px wide and saved at twice that, and the social card into
`assets/readme/social.png`. Its components live in `landing.py` and `landing.css` and use the
tokens and typefaces above.

The same command fills every block of `README.md` that sits between a pair of markers,
`<!-- name:start -->` and `<!-- name:end -->`. Those blocks hold every number on the page, and
each is built from `results/` and the chapter plans. Run the command after any chapter is
rebuilt. `tests/test_readme.py` fails when a block is stale.
