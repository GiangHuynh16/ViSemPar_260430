## Current saved results in `results/summary/all_scores.csv` 

Cover **Vietnamese public test** and **English Little Prince**. No Portuguese or Vietnamese v2 results are present in the summary CSV.

**Vietnamese**

| Method |      Smatch Mean |    Std | Rank |
| ------ | ---------------: | -----: | ---: |
| MTUP2  | **0.6494** | 0.0028 |    1 |
| B2     |           0.6466 | 0.0047 |    2 |
| MTUP3A |           0.6354 | 0.0040 |    3 |
| MTUP4  |           0.5970 | 0.0132 |    4 |
| B1     |           0.5917 | 0.0036 |    5 |
| MTUP3B |           0.5891 | 0.0084 |    6 |

Takeaway: `MTUP2` is best, but only slightly above `B2` by `+0.0028` absolute Smatch. Larger decompositions do not help.

**English Little Prince**

| Method |      Smatch Mean |    Std | Rank |
| ------ | ---------------: | -----: | ---: |
| B1     | **0.6747** | 0.0066 |    1 |
| B2     |           0.6647 | 0.0039 |    2 |
| MTUP2  |           0.6598 | 0.0141 |    3 |
| MTUP3A |           0.6556 | 0.0173 |    4 |
| MTUP3B |           0.6527 | 0.0052 |    5 |
| MTUP4  |           0.6410 | 0.0083 |    6 |

Takeaway: direct generation `B1` is best on English. MTUP variants underperform both baselines.

Across both datasets, parsable rate is `1.0000` for every method, so differences are mostly semantic/structural quality rather than basic AMR validity.

---

## Available AMR Benchmark Datasets
Most standard AMR parsing papers use:

| Dataset                                             | Language                           | Notes                                                                                                                                                                                                     |
| --------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AMR 1.0 / LDC2014T12**                      | English                            | Older benchmark, smaller; still cited historically.                                                                                                                                                       |
| **AMR 2.0 / LDC2017T10**                      | English                            | Very common benchmark.`36,521` train, `1,368` dev, `1,371` test, total `39,260`. Source: [LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10).                                                     |
| **AMR 3.0 / LDC2020T02**                      | English                            | Current major English benchmark.`55,635` train, `1,722` dev, `1,898` test, total `59,255`. Source: [LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02).                                           |
| **Chinese AMR 2.0 / LDC2021T13**              | Mandarin Chinese                   | About `20,078` Chinese sentences with train/dev/test splits. Source: [LDC2021T13](https://catalog.ldc.upenn.edu/LDC2021T13).                                                                               |
| **AMR 2.0 Four Translations / LDC2020T07**    | Spanish, German, Italian, Mandarin | Translations of AMR 2.0 test sentences, often used for cross-lingual parsing. Source:[LDC2020T07](https://catalog.ldc.upenn.edu/LDC2020T07).                                                                 |
| **AMR 3.0 Machine Translations / LDC2024T11** | Spanish, Irish, Dutch              | Machine-translated subset of AMR 3.0, useful for cross-lingual experiments. Source:[LDC2024T11](https://catalog.ldc.upenn.edu/LDC2024T11).                                                                   |
| **Little Prince AMR corpora**                 | Multiple languages                 | Small literary-domain corpora; useful for low-resource/cross-lingual analysis, but not the main SOTA benchmark.                                                                                           |
| **MASSIVE-AMR**                               | 50+ languages                      | Large multilingual AMR-style dataset over information-seeking utterances; not the same domain as classic AMR 2.0/3.0. Source:[Amazon MASSIVE-AMR](https://www.amazon.science/code-and-datasets/massive-amr). |

**Practical Recommendation**
If you want SOTA-comparable English results, use **AMR 2.0** or **AMR 3.0** official splits. If your paper is about low-resource/multilingual MTUP behavior, then Little Prince/Vietnamese can still be useful, but you should clearly state that the results are **not directly comparable to standard AMR SOTA tables**.

---

## Compare to available Little Prince results: Not exactly apples-to-apples.

Your current English Little Prince result is on a **custom split** in this repo: train on part of Little Prince, test on `143` held-out examples. Standard Little Prince AMR benchmark results usually evaluate on the **full TLP corpus**, about `1,562` examples, often as **out-of-distribution** evaluation after training on AMR 2.0/3.0.

**Your Best Result**

| Dataset                          | Best Method |     Smatch |
| -------------------------------- | ----------: | ---------: |
| English Little Prince repo split |      `B1` | `0.6747` |
| Best MTUP variant                |   `MTUP2` | `0.6598` |

**Published Little Prince / TLP Results**

| System              | Setting           | TLP Smatch |
| ------------------- | ----------------- | ---------: |
| SPRING DFS          | AMR2-trained OOD  |   `77.3` |
| SPRING DFS + silver | AMR2-trained OOD  |   `77.5` |
| AMRBART large       | OOD TLP benchmark |   `79.8` |
| LeakDistill         | OOD TLP benchmark |   `82.6` |
| CHAP + 200K         | OOD TLP benchmark |   `82.7` |

Sources: [SPRING paper](https://ojs.aaai.org/index.php/AAAI/article/download/17489/17296), [AMRBART / Graph Pre-training](https://aclanthology.org/2022.acl-long.415.pdf), [CHAP paper](https://aclanthology.org/2023.emnlp-main.553.pdf), [Papers with Code AMRBART TLP entry](https://paperswithcode.com/paper/graph-pre-training-for-amr-parsing-and-1).

There is also a Zensols/amrlib setup reporting `82.7` Smatch with `zsl_xfm_bart_large`, but that uses Little Prince/Bio data in training and reports development Smatch, so I would not treat it as the clean public TLP SOTA benchmark. Source: [zensols.amr docs](https://plandes.github.io/amr/index.html).

**Comparison**
Your `B1 = 67.47` is roughly:

- `-9.8` points behind SPRING `77.3`
- `-12.3` points behind AMRBART `79.8`
- `-15.2` points behind CHAP / Zensols-style `82.7`

So the current Little Prince result is **well below specialized AMR parsers**. It is closer to “reasonable small LoRA/decoder-only baseline” than to SOTA. Also, because your setup trains on Little Prince and still gets `~67`, while OOD parsers get `~77-83`, this suggests the bottleneck is likely model/training formulation/linearization quality, not just dataset size.

---

## Compare to equivalent setup

I don’t find an exact published match for:

`Qwen2.5-7B-Instruct + LoRA + train/test split inside English Little Prince AMR`.

So your current English LPP setup is not directly comparable to a known standard table. The closest comparisons fall into three buckets.

**Closest Comparisons**

| Comparison                  | Setup                                            |                                                                    Result | How Comparable                                             |
| --------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------: | ---------------------------------------------------------- |
| Your repo, English LPP      | Qwen2.5-7B + LoRA, in-domain LPP split           |                                           `B1 0.6747`, `MTUP2 0.6598` | This is the exact target setup.                            |
| Ettinger et al. 2023        | GPT-4 few-shot, small mixed sample including LPP |         LPP:`0.42` no fixes, `0.47` auto fixes, `0.51` manual fixes | Prompt-only, tiny sample, but directly includes LPP.       |
| Li & Fowlie 2025            | GPT-3.5/GPT-4o prompting, 0-shot/few-shot/CoT    |                                    best about `0.60` with GPT-4o 5-shot | Prompt-only, not Qwen/LoRA; useful lower-bound comparison. |
| Ho 2025                     | decoder-only LLMs + LoRA on AMR 3.0              |                                                  LLaMA 3.2 best `0.804` | Decoder-only LoRA, but large AMR 3.0 split, not LPP.       |
| VLSP 2025 Vietnamese papers | Qwen3/Llama/Gemma/Phi + LoRA/SFT                 | Qwen3-14B:`0.54` public, `0.58` private; Qwen3-1.7B: `0.46` private | Very comparable to your Vietnamese setup, not English LPP. |

**What This Means**
Your English LPP `0.6747` is:

- clearly better than published **prompt-only GPT** AMR parsing results on/near LPP,
- below specialized AMR parsers on LPP, which are often around `0.75-0.83`,
- below decoder-only LoRA results on large AMR 3.0, e.g. `0.804`, but that uses much more training data and a standard English benchmark.

The most relevant published few-shot LPP number I found is from Ettinger et al. They report on a small sample including LPP, where GPT-4 few-shot gets LPP Smatch `0.42`, `0.47` with automatic fixes, and `0.51` with manual formatting fixes; AMRBART gets `0.75` on the same LPP sample. Source: [Ettinger et al. 2023](https://aclanthology.org/2023.findings-emnlp.553.pdf).

For broader prompt-only AMR parsing, Li & Fowlie report GPT-4o maxing around `0.60` Smatch in a 5-shot setting. Source: [GPT makes a poor AMR parser](https://jlcl.org/article/view/285).

For decoder-only LoRA, Ho 2025 is the closest architectural comparison, but it uses AMR 3.0, not LPP. It reports LLaMA 3.2 reaching `0.804` Smatch on LDC2020T02 AMR 3.0. Source: [arXiv 2508.05028](https://arxiv.org/abs/2508.05028).

For Vietnamese decoder-only/Qwen-style systems, VLSP 2025 papers are quite relevant: one reports Qwen3-14B with LoRA reaching `0.54` public and `0.58` private Smatch; another Qwen3-1.7B system reports `0.46` private. Sources: [VLSP empirical study](https://aclanthology.org/2025.vlsp-1.34.pdf), [ViAMR](https://aclanthology.org/2025.vlsp-1.31.pdf).

Bottom line: **your English LPP result seems strong relative to prompt-only LLMs, but weak relative to specialized AMR parsers. I did not find a clean published Qwen2.5-7B LoRA + within-LPP split baseline.**
