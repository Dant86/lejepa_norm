# Architecture

Update these diagrams in the same PR as any change that alters module dependencies
or the flow between apps.

## Module dependencies

Arrows point from a module to what it imports.

```mermaid
graph LR
    subgraph apps
        PD["apps/prepare_data.py"]
        TR["apps/train.py"]
        EM["apps/embed.py"]
        AN["apps/analyze.py"]
    end
    subgraph lejepa_norm
        CFG["config"]
        PTH["paths"]
        DATA["data"]
        MOD["models"]
        LOSS["losses"]
        TRAIN["train"]
        EVAL["eval"]
        PLOT["plotting"]
        REP["reporting"]
    end
    PD --> CFG
    PD --> PTH
    PD --> DATA
    TR --> CFG
    TR --> PTH
    TR --> TRAIN
    EM --> CFG
    EM --> PTH
    EM --> EVAL
    AN --> CFG
    AN --> PTH
    AN --> EVAL
    AN --> PLOT
    AN --> REP
    TRAIN --> CFG
    TRAIN --> DATA
    TRAIN --> MOD
    TRAIN --> LOSS
    TRAIN --> EVAL
    EVAL --> DATA
    EVAL --> MOD
    EVAL --> CFG
    DATA --> CFG
    MOD --> CFG
    LOSS --> CFG
    PLOT --> CFG
```

## Data and experiment flow

```mermaid
flowchart TD
    SRC["Canonical sources: STL-10, DTD, SVHN"] --> PD["apps/prepare_data.py"]
    PD --> DATA[("Prepared data: memmaps + STL-10-C")]
    DATA --> TR["apps/train.py (variant A, B or C)"]
    CFG["configs/*.yaml"] --> TR
    TR --> RUN[("Run dir: checkpoints, resolved config, metrics.jsonl")]
    RUN --> EM["apps/embed.py"]
    DATA --> EM
    EM --> EMB[("Embedding dumps: z, h, K views per image")]
    EMB --> AN["apps/analyze.py"]
    RUN --> AN
    AN --> TAB["Markdown tables"]
    AN --> PNG["PNG figures (Plotly + Kaleido)"]
    TAB --> NOTE["research_notes/IDEA_NAME/experiment.md"]
    PNG --> NOTE
```

Only `config` and `paths` exist as of M0; the remaining modules and apps land in
later milestones (see the Technical Design).
