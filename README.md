# Active Learning for Industrial Vision Systems

This repository contains controlled active-learning experiments for binary industrial defect classification using ResNet50 and ViT-B/16.

For each model, three strategies are maintained:

Baseline — trained only on the initial labeled training set.
Active Learning — adds 20 selected images per cycle.
Random Control — adds the same number of randomly selected images per cycle.

Two acquisition implementations are kept:

Old pipeline — randomly sample 40 candidates, score them using uncertainty, diversity, and novelty/drift, then select the top 20.
New pipeline — score the full unlabeled pool using uncertainty and novelty/drift, keep the top 40 informative samples, then apply greedy k-center to select 20 diverse samples.

This README documents the repository layout and how to reproduce the experiments.

**Important execution rule:** the repository structure stays exactly as shown below.

The old and new pipelines are kept as separate experiment branches so the original files, outputs, and results remain visible.

**Run each pipeline as a complete independent experiment from Cycle 0 to Cycle 5. Do not switch from the old pipeline to the new pipeline halfway through a run.**

---

## 1. Repository Structure

```text

aachen_project/
│
├── README.md
├── split.json
├── raw_data/
│   ├── defect/
│   ├── no_defect/
│   ├── extra/
│   │   └──dark/
│   └── unlabeled_pool/
│
├── src/
│   ├── resnet/
│   │   ├── old_pipeline/
│   │   └── new_pipeline/
│   └── vit/
│       ├── old_pipeline/
│       └── new_pipeline/
│
├── checkpoints/
│   ├── resnet/
│   │   ├── old_pipeline/{baseline,active,random}/
│   │   └── new_pipeline/{baseline,active,random}/
│   └── vit/
│       ├── old_pipeline/{baseline,active,random}/
│       └── new_pipeline/{baseline,active,random}/
│
├── logs/
│   ├── resnet/{old_pipeline,new_pipeline}/
│   └── vit/{old_pipeline,new_pipeline}/
│
├── results/
│   ├── resnet/{old_pipeline,new_pipeline}/
│   └── vit/{old_pipeline,new_pipeline}/
│
├── selections/
│   ├── resnet/
│   │   ├── old_pipeline/{active,random}/
│   │   └── new_pipeline/{active,random}/
│   └── vit/
│       ├── old_pipeline/{active,random}/
│       └── new_pipeline/{active,random}/
│
└── states/
    ├── resnet/
    │   ├── old_pipeline/{active,random}/
    │   └── new_pipeline/{active,random}/
    └── vit/
        ├── old_pipeline/{active,random}/
        └── new_pipeline/{active,random}/

```

---

## 2. Dataset

The binary labels are:

```text

no_defect = 0

defect    = 1

```

The controlled split is stored in `split.json`.

Main experiment sizes:

| Split | Size |
|---|---:|
| Initial training set | 160 |
| Validation set | 40 |
| Test set | 40 |
| Shadow set | 100 |
| Initial unlabeled pool | 200 |

The split must remain unchanged during a controlled experiment.

The expected active-learning progression is:

| Cycle | Labeled train | Remaining pool |
|---|---:|---:|
| C0 | 160 | 200 |
| C1 | 180 | 180 |
| C2 | 200 | 160 |
| C3 | 220 | 140 |
| C4 | 240 | 120 |
| C5 | 260 | 100 |

---

## 3. Important Generated Files

### States

Each state records the current labeled training set and remaining pool:

```text

cycle_00.json
cycle_01.json

...

cycle_05.json

```

A state typically contains:

```json

{

  "cycle": 1,

  "labeled_train": [

    {

      "path": "defect/example.BMP",

      "label": 1,

      "added_cycle": 0

    }

  ],

  "unlabeled_pool": [],

  "selected_history": []

}

```

### Selections

#### Old pipeline

Typical active-learning outputs:

```text

cycle_01_candidates.csv
cycle_01_to_label.csv
cycle_01_embeddings.npz

```

#### New pipeline

The full-pool implementation also saves:

```text

cycle_01_pool_scores.csv

```

So a new-pipeline selection cycle usually produces:

```text

cycle_01_pool_scores.csv
cycle_01_candidates.csv
cycle_01_to_label.csv
cycle_01_embeddings.npz

```

### Checkpoints

Recommended naming:

```text

baseline/best.pt
active/cycle_01.pt

...

active/cycle_05.pt
random/cycle_01.pt

...

random/cycle_05.pt

```

### Logs

Typical training logs:

```text

baseline/train_log.csv
active/cycle_01_train_log.csv

...

random/cycle_05_train_log.csv

```

### Results

Evaluation outputs should be stored under the correct model and pipeline, for example:

```text

results/resnet/old_pipeline/
results/resnet/new_pipeline/
results/vit/old_pipeline/
results/vit/new_pipeline/

```

The separate `old_pipeline/` and `new_pipeline/` directories are intentional. They preserve the files and outputs from both implementations and make it possible to present or inspect each experiment independently. Do not merge these directories together.

---

## 4. Environment Setup

Run commands from the repository root.

Windows example:

```powershell

cd C:\Users\<username>\Documents\aachen_project

```

Create a virtual environment:

```powershell

python -m venv .venv

.venv\Scripts\Activate.ps1

```

Linux/macOS:

```bash

python3 -m venv .venv

source .venv/bin/activate

```

Install dependencies:

```powershell

pip install torch torchvision numpy pillow scikit-learn matplotlib

```

If a `requirements.txt` exists:

```powershell

pip install -r requirements.txt

```

Check CUDA:

```powershell

python -c "import torch; print(torch.cuda.is_available())"

```

The scripts can run on CPU, but training will be slower.

---

## 5. Old vs New Acquisition Pipeline

### Old pipeline

```text

Current pool

    |

Randomly sample 40

    |

Run current model

    |

    +-- uncertainty: normalized predictive entropy

    +-- diversity: nearest-neighbour embedding distance

    +-- novelty/drift: distance from Cycle-0 training embeddings

    |

Normalize diversity and novelty

    |

Equal-weight combined score

    |

Rank 40

    |

Select top 20

```

Only the selected 20 are removed from the pool. The rejected 20 remain available for later cycles.

### New pipeline

```text

Entire current pool

    |

Score every image

    |

    +-- uncertainty

    +-- novelty/drift

    |

Information score

    |

Keep top 40

    |

Greedy k-center

    |

Select 20

```

Default information score:

```text

0.5 * uncertainty + 0.5 * normalized novelty

```

The new pipeline removes the random-40 bottleneck before informative scoring.

---

## 6. Manual Labeling Step

After selection, a file such as:

```text

cycle_01_to_label.csv

```

is created.

Before labeling:

```csv

filename,label
unlabeled_pool/image_001.BMP,
unlabeled_pool/image_002.BMP,

```

Fill labels manually:

```csv

filename,label
unlabeled_pool/image_001.BMP,0
unlabeled_pool/image_002.BMP,1

```

Use:

```text

0 = no_defect
1 = defect

```

Do not run the state-update script until all 20 selected images have valid labels.

---

## 7. Complete Experiment Order

Each model/pipeline combination must be run as a separate complete experiment.

For one experiment branch:

```text

1. Verify split.json
2. Initialize that branch's Active + Random Cycle-0 states
3. Train that branch's Baseline
4. Run Active C1-C5 using only that pipeline's selection method
5. Run that branch's Random C1-C5
6. Evaluate that branch's Baseline
7. Evaluate that branch's Active C5
8. Evaluate that branch's Random C5
9. Optionally run integrity audit
10. Optionally compare per-image predictions
11. Optionally generate training graphs

```

The four independent experiment families are:

```text

Experiment 1: ResNet + old pipeline
Experiment 2: ResNet + new pipeline
Experiment 3: ViT    + old pipeline
Experiment 4: ViT    + new pipeline

```

The second run must not continue from the first run's Cycle-5 state. Both experiments should start from the same fixed `split.json`, but their states, selections, checkpoints, logs, and results remain in their own existing pipeline directories.

---

## 8. ResNet State Initialization

The cleaned initializer should create both Active and Random Cycle-0 states from the same `split.json`.

If stored inside the old pipeline:

```powershell

python src/resnet/init_resnet_state.py --pipeline old_pipeline

```

For an isolated new-pipeline experiment:

```powershell

python src/resnet/init_resnet_state.py --pipeline new_pipeline

```

If the initializer is shared:

```powershell

python src/resnet/init_resnet_state.py --pipeline <old_pipeline|new_pipeline>

```

Run it once per experiment.

Both branches must start with identical:

```text

160 labeled training images
200 unlabeled images

```

---

## 9. ResNet Baseline

Train the baseline before any acquisition cycle.

Example:

```powershell

python src/resnet/train_baseline.py --pipeline old_pipeline

```

or, if named explicitly:

```powershell

python src/resnet/train_baseline.py --pipeline old_pipeline

```

For the new pipeline:

```powershell

python src/resnet/train_baseline.py --pipeline new_pipeline

```

The resulting baseline checkpoint is used for Cycle-1 Active selection.

---

## 10. ResNet Old Active Pipeline

### Cycle 1

Selection:

```powershell

python src/resnet/old_pipeline/selection.py --cycle 1 --checkpoint checkpoints/resnet/old_pipeline/baseline/best.pt

```

Manually label:

```text

selections/resnet/old_pipeline/active/cycle_01_to_label.csv

```

Update state:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 1 --pipeline old_pipeline

```

Train:

```powershell

python src/resnet/train.py --cycle 1 --pipeline old_pipeline

```

### Cycle 2

```powershell

python src/resnet/old_pipeline/selection.py --cycle 2 --checkpoint checkpoints/resnet/old_pipeline/active/cycle_01.pt

```

Label `cycle_02_to_label.csv`, then:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 2 --pipeline old_pipeline
python src/resnet/train.py --cycle 2 --pipeline old_pipeline

```

### Cycle 3

```powershell

python src/resnet/old_pipeline/selection.py --cycle 3 --checkpoint checkpoints/resnet/old_pipeline/active/cycle_02.pt

```

Label, then:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 3 --pipeline old_pipeline
python src/resnet/train.py --cycle 3 --pipeline old_pipeline

```

### Cycle 4

```powershell

python src/resnet/old_pipeline/selection.py --cycle 4 --checkpoint checkpoints/resnet/old_pipeline/active/cycle_03.pt

```

Label, then:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 4 --pipeline old_pipeline
python src/resnet/train.py --cycle 4 --pipeline old_pipeline

```

### Cycle 5

```powershell

python src/resnet/old_pipeline/selection.py --cycle 5 --checkpoint checkpoints/resnet/old_pipeline/active/cycle_04.pt

```

Label, then:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 5 --pipeline old_pipeline
python src/resnet/train.py --cycle 5 --pipeline old_pipeline

```

Checkpoint dependency:

| Selection | Checkpoint |
|---|---|
| C1 | Baseline |
| C2 | Active C1 |
| C3 | Active C2 |
| C4 | Active C3 |
| C5 | Active C4 |

---

## 11. ResNet New Active Pipeline

The workflow is the same, but use the full-pool + k-center selector.

This is a new independent experiment, not a continuation of the old-pipeline run. Initialize the new-pipeline Cycle-0 state from the same `split.json`, train its baseline, and then run Cycles 1-5 entirely inside the new-pipeline branch.

### Cycle 1:

```powershell

python src/resnet/new_pipeline/selection_resnet_new.py --cycle 1 --checkpoint checkpoints/resnet/new_pipeline/baseline/best.pt

```

Label:

```text

selections/resnet/new_pipeline/active/cycle_01_to_label.csv

```

Then:

```powershell

python src/resnet/old_pipeline/active_learning.py --cycle 1 --pipeline new_pipeline
python src/resnet/train.py --cycle 1 --pipeline new_pipeline

```

### Cycle 2:

```powershell

python src/resnet/new_pipeline/selection_resnet_new.py --cycle 2 --checkpoint checkpoints/resnet/new_pipeline/active/cycle_01.pt
python src/resnet/old_pipeline/active_learning.py --cycle 2 --pipeline new_pipeline
python src/resnet/train.py --cycle 2 --pipeline new_pipeline

```

### Cycle 3:

```powershell

python src/resnet/new_pipeline/selection_resnet_new.py --cycle 3 --checkpoint checkpoints/resnet/new_pipeline/active/cycle_02.pt
python src/resnet/old_pipeline/active_learning.py --cycle 3 --pipeline new_pipeline
python src/resnet/train.py --cycle 3 --pipeline new_pipeline

```

### Cycle 4:

```powershell

python src/resnet/new_pipeline/selection_resnet_new.py --cycle 4 --checkpoint checkpoints/resnet/new_pipeline/active/cycle_03.pt
python src/resnet/old_pipeline/active_learning.py --cycle 4 --pipeline new_pipeline
python src/resnet/train.py --cycle 4 --pipeline new_pipeline

```

### Cycle 5:

```powershell

python src/resnet/new_pipeline/selection_resnet_new.py --cycle 5 --checkpoint checkpoints/resnet/new_pipeline/active/cycle_04.pt
python src/resnet/old_pipeline/active_learning.py --cycle 5 --pipeline new_pipeline
python src/resnet/train.py --cycle 5 --pipeline new_pipeline

```

Remember to fill each `cycle_XX_to_label.csv` before its state update.

---

## 12. ResNet Random Control

Each Random cycle follows:

```text

random selection
-> manual labeling
-> random state update
-> training

```

### Cycle 1:

```powershell

python src/resnet/random_selection.py --cycle 1 --pipeline old_pipeline

```

Label:

```text

selections/resnet/old_pipeline/random/cycle_01_to_label.csv

```

Then:

```powershell

python src/resnet/update_random_state.py --cycle 1 --pipeline old_pipeline
python src/resnet/train_random.py --cycle 1 --pipeline old_pipeline

```

Repeat for cycles 2-5:

```powershell

python src/resnet/random_selection.py --cycle 2 --pipeline old_pipeline
python src/resnet/update_random_state.py --cycle 2 --pipeline old_pipeline
python src/resnet/train_random.py --cycle 2 --pipeline old_pipeline

```

and so on.

For the new-pipeline experiment, run the equivalent Random branch under `new_pipeline/` from its own fresh Cycle-0 state.
Do not reuse the old-pipeline Random Cycle-1 to Cycle-5 states or checkpoints when reporting the new-pipeline experiment.

---

## 13. ViT Initialization

Initialize ViT Cycle-0 Active and Random states:

```powershell

python src/vit/init_vit_state.py --pipeline old_pipeline

```

or:

```powershell

python src/vit/init_vit_state.py --pipeline new_pipeline

```

Both states must come from the same `split.json`.

---

## 14. ViT Baseline

For the FT2 configuration:

```powershell

python src/vit/train_baseline_vit_ft2.py --epochs 10 --pipeline old_pipeline

```

New pipeline:

```powershell

python src/vit/train_baseline_vit_ft2.py --epochs 10 --pipeline new_pipeline

```

Use the same number of epochs for Baseline, Active, and Random within the same controlled comparison.

---

## 15. ViT Old Active Pipeline

### Cycle 1:

```powershell

python src/vit/old_pipeline/selection_vit.py --cycle 1 --checkpoint checkpoints/vit/old_pipeline/baseline/best.pt

```

Label the generated file, then:

```powershell

python src/vit/old_pipeline/active_learning_vit.py --cycle 1 --pipeline old_pipeline
python src/vit/train_vit_ft2.py --cycle 1 --epochs 10 --pipeline old_pipeline

```

### Cycle 2:

```powershell

python src/vit/old_pipeline/selection_vit.py --cycle 2 --checkpoint checkpoints/vit/old_pipeline/active/cycle_01.pt
python src/vit/old_pipeline/active_learning_vit.py --cycle 2 --pipeline old_pipeline
python src/vit/train_vit_ft2.py --cycle 2 --epochs 10 --pipeline old_pipeline

```

Continue through Cycle 5, always using the previous Active checkpoint.

---

## 16. ViT New Active Pipeline

This is a separate ViT experiment from the old-pipeline run. Start again from the new-pipeline Cycle-0 state created from the same fixed `split.json`, retrain the baseline for that experiment, and use the new selector consistently through Cycle 5.

### Cycle 1:

```powershell

python src/vit/new_pipeline/selection_vit_new.py --cycle 1 --checkpoint checkpoints/vit/new_pipeline/baseline/best.pt

```

Label the generated `cycle_01_to_label.csv`, then:

```powershell

python src/vit/old_pipeline/active_learning_vit.py --cycle 1 --pipeline new_pipeline
python src/vit/train_vit_ft2.py --cycle 1 --epochs 10 --pipeline new_pipeline

```

### Cycle 2:

```powershell

python src/vit/new_pipeline/selection_vit_new.py --cycle 2 --checkpoint checkpoints/vit/new_pipeline/active/cycle_01.pt
python src/vit/old_pipeline/active_learning_vit.py --cycle 2 --pipeline new_pipeline
python src/vit/train_vit_ft2.py --cycle 2 --epochs 10 --pipeline new_pipeline

```

Repeat through Cycle 5.
The new selector should additionally generate a `cycle_XX_pool_scores.csv` proving that every image in the current pool was scored.

---

## 17. ViT Random Control

### Cycle 1:

```powershell

python src/vit/random_selection_vit.py --cycle 1 --pipeline old_pipeline

```

Label:

```text

selections/vit/old_pipeline/random/cycle_01_to_label.csv

```

Then:

```powershell

python src/vit/update_random_state_vit.py --cycle 1 --pipeline old_pipeline
python src/vit/train_random_vit_ft2.py --cycle 1 --epochs 10 --pipeline old_pipeline

```

Repeat for Cycles 2-5.

For the new-pipeline experiment, use the corresponding `new_pipeline/` Random paths and run that Random branch from a fresh Cycle-0 state.
Keep the old- and new-pipeline Random results separate so each Active-vs-Random comparison belongs to the same independent experiment branch.

---

## 18. Evaluation

Final models to evaluate:

```text

Baseline
Active C5
Random C5

```

### ResNet

```powershell

python src/resnet/evaluate.py --pipeline old_pipeline --checkpoint checkpoints/resnet/old_pipeline/baseline/best.pt --split test --name baseline_test
python src/resnet/evaluate.py --pipeline old_pipeline --checkpoint checkpoints/resnet/old_pipeline/active/cycle_05.pt --split test --name active_c5_test
python src/resnet/evaluate.py --pipeline old_pipeline --checkpoint checkpoints/resnet/old_pipeline/random/cycle_05.pt --split test --name random_c5_test

```

Repeat with:

```text

\--split shadow

```

### ViT

```powershell

python src/vit/evaluate_vit.py --pipeline old_pipeline --checkpoint checkpoints/vit/old_pipeline/baseline/best.pt --split test --name vit_baseline_test
python src/vit/evaluate_vit.py --pipeline old_pipeline --checkpoint checkpoints/vit/old_pipeline/active/cycle_05.pt --split test --name vit_active_c5_test
python src/vit/evaluate_vit.py --pipeline old_pipeline --checkpoint checkpoints/vit/old_pipeline/random/cycle_05.pt --split test --name vit_random_c5_test

```

Again, repeat with `--split shadow`.
Use equivalent paths for the new pipeline.
When reporting results, compare models within the same experiment branch:

```text

old-pipeline Baseline vs old-pipeline Active vs old-pipeline Random
new-pipeline Baseline vs new-pipeline Active vs new-pipeline Random

```

## 19. Prediction Error Comparison (Optional)

`compare_predictions.py` can be run to compare Baseline, Active, and Random image by image.

Example:

```powershell

python src/compare_predictions.py `
  --root . `
  --architecture resnet50 `
  --model baseline=checkpoints/resnet/old_pipeline/baseline/best.pt `
  --model active=checkpoints/resnet/old_pipeline/active/cycle_05.pt `
  --model random=checkpoints/resnet/old_pipeline/random/cycle_05.pt `
  --split test `
  --split shadow `
  --out results/resnet/old_pipeline/prediction_comparison

```

Typical outputs:

```text

test_paired.csv
test_errors.csv
test_false_negatives.csv
shadow_paired.csv
shadow_errors.csv
shadow_false_negatives.csv
metrics.csv
manifest.json

```

This is a read-only analysis step.

---

## 20. Training Graphs (Optional)

```powershell

python src/analyze_training.py `
  --root . `
  --run baseline=logs/resnet/old_pipeline/baseline/train_log.csv `
  --run active=logs/resnet/old_pipeline/active/cycle_05_train_log.csv `
  --run random=logs/resnet/old_pipeline/random/cycle_05_train_log.csv `
  --output results/resnet/old_pipeline/training_analysis

```

Typical outputs:

```text

baseline_loss.png
baseline_accuracy.png
active_loss.png
active_accuracy.png
random_loss.png
random_accuracy.png
comparison_val_loss.png
comparison_val_acc.png
summary.csv
all_epochs.csv
report.json

```

---

## 21. Integrity Audit (Optional)

Run the integrity audit before finalizing an experiment.

Example:

```powershell

python src/audit_integrity.py `
  --root .
  --near-duplicates

```

The audit is intended to check:

```text

Cycle-0 branch equality
state progression
train/pool conservation
selection/state agreement
label consistency
held-out leakage
missing image files
exact duplicates
possible near duplicates

```

Near-duplicate warnings require manual review and are not automatically proof of leakage.

---

## 22. Restarting a Run or Running the Other Pipeline

The existing directory structure already keeps old- and new-pipeline files separate, so do not delete or merge the completed branch just to run the other pipeline.

If a run itself must be restarted:

- identify the exact model/pipeline branch;
- preserve any outputs that should be kept;
- remove only that branch's generated states, selections, checkpoints, logs, and results;
- rerun that branch's Cycle-0 initialization;
- retrain that branch's baseline;
- rerun acquisition from Cycle 1.

If you want to run the other pipeline, treat it as a new experiment:

Both can remain in the repository at the same time because their files and outputs are already stored in separate `old_pipeline/` and `new_pipeline/` directories.

Do not reuse later-cycle states after changing:

```text

labels
selection algorithm
split.json
model configuration
training settings
seed policy

```

because later selections depend on earlier checkpoints.
