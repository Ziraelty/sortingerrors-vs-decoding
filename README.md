# Spike-sorting errors vs. decoding (NLB MC_Maze)

I wanted to see how different kinds of spike-sorting errors affect a decoder that was trained on clean data. Short answer: errors that mix up which neuron a unit is (label swaps, merges) hurt much more than errors that just remove or add spikes. Swapping labels for 25% of units dropped R² from 0.70 to 0.34 even though no spikes were lost, which is worse than deleting half the units (0.40). Splits barely mattered.

<img width="690" height="440" alt="image" src="https://github.com/user-attachments/assets/f4cae5cc-9e21-49ed-82e0-25b6c27d04c7" />


(mean ± SD over 5 random seeds)

## Why

Earlier work found that sorting quality doesn't matter much for decoding when the decoder is retrained afterwards ([Christie et al. 2015](https://escholarship.org/uc/item/0ww0z6xm), [Trautmann et al. 2019](https://www.biorxiv.org/content/10.1101/229252v2.full)). But newer models like POYO learn an embedding for each unit, and [BrainWideBench](https://arxiv.org/html/2609.22064) mentions that shifting probes affect sorting quality over time. So I wanted to know what happens when the sorting changes after the decoder has already been calibrated.

## Setup

- Data: NLB MC_Maze, `jenkins_maze_train` session, loaded with torch_brain. 142 units, 10 ms bins, 75 train / 25 val trials, predicting 2D hand velocity.
- Model: the bidirectional GRU from the torch_brain NLB Maze tutorial, trained once on clean data (100 epochs, ~18 s on a Colab T4).
- Then I corrupted only the validation spike counts (same units affected in every trial) and measured R². 6 error types x 4 levels x 5 seeds.

Error types:

- `dropout`: chosen units go silent
- `missed`: every unit loses a fraction of its spikes
- `split`: half of a chosen unit's spikes move to a unit the decoder never saw
- `merge`: pairs of units are combined, one of them goes silent
- `false_pos`: Poisson noise spikes added to all units
- `swap`: chosen units swap labels

For dropout, split, merge and swap the level is the fraction of units affected. For missed it's the fraction of spikes removed, and for false_pos the noise rate as a fraction of the mean firing rate.

## Results (mean R²)

| error | 0% | 10% | 25% | 50% |
| --- | --- | --- | --- | --- |
| dropout | 0.700 | 0.683 | 0.593 | 0.395 |
| missed | 0.700 | 0.689 | 0.643 | 0.503 |
| split | 0.700 | 0.697 | 0.680 | 0.629 |
| merge | 0.700 | 0.667 | 0.567 | 0.384 |
| false_pos | 0.700 | 0.684 | 0.610 | 0.496 |
| swap | 0.700 | 0.580 | 0.335 | 0.023 |

## Notes

- Swaps don't remove any spikes, so the whole drop comes from the decoder not knowing which unit is which. Retraining on swapped data would be the same as retraining on clean data, which fits with the earlier results.
- Merges are worse than just dropping units. At 50%, about 25% of units go silent and another 25% get mixed with a different neuron. Dropping 25% alone gives 0.59, merging gives 0.38.
- Splits, missed spikes and false positives degrade more slowly because the remaining spikes still come from the right units.

## Limitations

- one session, one monkey, one model (not POYO)
- the GRU is bidirectional, so this is offline decoding
- only 25 validation trials and 5 seeds; swap results vary a lot depending on which units get swapped
- the corruptions are random, real sorting errors aren't (merges usually happen between nearby, similar-looking units)

## Next

- try the same with POYO: does re-learning unit embeddings fix identity errors, and how many trials does it need?
- more realistic errors (swaps between similar units, BrainWideBench QC metadata)
- more sessions and a causal decoder

## Running it

Colab with a T4 GPU. First cell:

```
!pip install -q scikit-learn matplotlib pandas dandi
!pip install -q git+https://github.com/neuro-galaxy/torch_brain
!brainsets prepare pei_pandarinath_nlb_2021 --raw-dir data/raw --processed-dir data/processed -s jenkins_maze_train
```

Then paste `sorting_errors.py` into a second cell. It prints the clean R² and the results table, and saves `results.csv` and `sorting_errors.png`.

Locally: `pip install -r requirements.txt`, run the brainsets command above, then `python sorting_errors.py`.

The dataset class and GRU are adapted from the [torch_brain](https://github.com/neuro-galaxy/torch_brain) [NLB Maze tutorial](https://torch-brain.readthedocs.io/en/latest/generated/notebooks/nlb_maze_minimal_example.html).
