This folder contains the input data format for `ScenarioLoaderMIMICIV`. Scenario loader format can be adapted for other data sources.

Scenario loader takes in as input:
- `OSCE-format.jsonl`, an OSCE scenario for the agent to play out. Can be made so the agent only receives a clinical note to process.
- `original-notes.txt`, the original clinical note the OSCE scenario is based off of.

Because this is MIMIC-IV data, only credentialed users can contact us to obtain the complete `OSCE-format.jsonl` and `original-notes.txt` data. (https://physionet.org/content/mimiciv/3.1/)
