# Schema authority

Authoritative JSON Schemas for the Jynx protocol belong in the **RULES** repository ([jynx-rules](https://github.com/HNXJ/jynx-rules)).

| Schema | `$id` | Authoritative repo | Copy in jynx-engine |
|--------|-------|--------------------|--------------------|
| Tournament | `https://jynx.dev/schemas/Tournament.schema.json` | jynx-rules | provisional |
| RoundManifest | `https://jynx.dev/schemas/RoundManifest.schema.json` | jynx-rules | provisional |
| Move | `https://jynx.dev/schemas/Move.schema.json` | jynx-rules | provisional |
| EvidenceReceipt | `https://jynx.dev/schemas/EvidenceReceipt.schema.json` | jynx-rules | provisional |
| Judgment | `https://jynx.dev/schemas/Judgment.schema.json` | jynx-rules | provisional |
| Common types | `https://jynx.dev/schemas/common/*.schema.json` | jynx-rules | provisional |

Until migration, changes land in jynx-engine first for review, then are promoted to jynx-rules without altering runtime behavior in this repository.

Fixtures under `schemas/fixtures/` are **non-normative** examples for tests.
