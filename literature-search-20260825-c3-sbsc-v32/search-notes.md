# Search Notes

Date: 2026-08-25  
Mode: standard  
Purpose: prior-art and novelty-risk search for the proposed C³-SBSC V3.2 evidence-routing operator before architecture freeze.

## Safe Queries Used

Only public technical concepts, public paper titles, venue names and dataset names were queried. No unpublished result values, draft sentences, source code or local paths were placed in web queries.

- `SCTransNet spatial-channel cross transformer infrared small target detection official`
- `infrared small target detection false alarm adversarial learning official CVF`
- `infrared small target clutter suppression false alarm CVPR WACV`
- `DNANet dense nested attention infrared small target IEEE TIP`
- `ISNet Shape Matters infrared small target CVPR 2022`
- `MSHNet scale location sensitivity CVPR 2024`
- `infrared small target distractor discriminator contrastive arXiv`
- `Seeing Through the Noise infrared small target CVPR 2026`
- `difficulty-aware segmentation error supervised attention`
- `DEAL difficulty-aware active learning semantic segmentation ACCV`
- `EAR-NET error attention refining segmentation IEEE`
- `false positive rectification semantic segmentation ICCV`
- `semantic segmentation reverse attention three branch BMVC`
- `learning dynamic routing semantic segmentation CVPR`
- `constrained attention KL projection simplex ACL`
- `sparse and structured neural attention NeurIPS`
- `Sparse Sinkhorn Attention ICML`
- `Auto Learning Attention NeurIPS`

## Sources Checked

- Primary proceedings/publisher records: CVF Open Access, ACL Anthology, NeurIPS proceedings, PMLR, BMVA archive, AAAI/OJS, IEEE DOI/Xplore and Springer publisher pages.
- Stable public records used where a proceedings page was unavailable: arXiv, DOI landing pages and an institutional accepted-manuscript record.
- Public project/dataset repositories were consulted only for code/data availability and naming cautions, not as evidence of performance superiority.

## Screening Ledger

Twenty-three unique candidates were screened after title deduplication. Fifteen were retained in `papers.md`/`papers.csv` because they cover at least one required axis and materially affect the claim boundary.

`I/C/N` denotes the 1–5 insight/completeness/numeric-evidence scores. These are paper-quality diagnostics, not acceptance probabilities.

| Candidate | Year | Venue/status | Primary URL | I/C/N | Evidence used for screening | Decision |
| --- | ---: | --- | --- | --- | --- | --- |
| SCTransNet | 2024 | IEEE TGRS | [DOI](https://doi.org/10.1109/TGRS.2024.3383649) | 4/4/4 | Same SCTB/SSCA host and full-scale cross-level interaction. | Include: exact architectural baseline. |
| MDvsFA | 2019 | ICCV | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.html) | 5/4/4 | Explicit adversarial decomposition of miss detection and false alarm. | Include: direct problem/claim risk. |
| ACM | 2021 | WACV | [CVF](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html) | 4/4/4 | Top-down/bottom-up contextual modulation and SIRST benchmark. | Include: cross-level and dataset anchor. |
| ALCNet | 2021 | IEEE TGRS | [DOI](https://doi.org/10.1109/TGRS.2020.3044958) | 4/4/4 | Local-contrast prior modularized inside an attentional network. | Include: static contrast/rarity risk. |
| DNANet | 2023 | IEEE TIP | [DOI](https://doi.org/10.1109/TIP.2022.3199107) | 4/5/5 | Dense cross-level attention, NUDT-SIRST and Pd/Fa definitions. | Include: benchmark and strong baseline. |
| ISNet | 2022 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html) | 4/5/5 | Shape/edge mechanism and IRSTD-1K benchmark. | Include: benchmark and alternative mechanism. |
| MSHNet/SLS loss | 2024 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | 4/5/5 | Scale/location-sensitive supervision with a simple multi-scale head. | Include: auxiliary-loss confounder. |
| NS-FPN | 2026 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | 4/4/4 | Frequency/noise suppression aimed at reducing IRSTD false alarms. | Include: recent direct competitor. |
| CCDNet | 2026 | arXiv preprint | [arXiv](https://arxiv.org/abs/2603.29228) | 4/3/3 | Local/global contrastive distractor discriminator. | Include with preprint caution: closest distractor framing. |
| DEAL | 2020 | ACCV | [CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Xie_DEAL_Difficulty-aware_Active_Learning_for_Semantic_Segmentation_ACCV_2020_paper.html) | 4/4/4 | Current prediction error supervises pixel-wise difficulty attention. | Include: direct supervision overlap. |
| EAR-NET | 2021 | DICTA | [DOI](https://doi.org/10.1109/DICTA52665.2021.9647299) | 4/3/3 | Prediction-versus-GT error map supervises attention/refinement. | Include: direct supervision overlap. |
| Semantic Segmentation with Reverse Attention | 2017 | BMVC | [BMVA](https://bmva-archive.org.uk/bmvc/2017/papers/paper018/paper018.pdf) | 4/4/4 | Direct, reverse and reverse-attention branches. | Include: opposite-role/three-branch risk. |
| Learning Dynamic Routing for Semantic Segmentation | 2020 | CVPR | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Learning_Dynamic_Routing_for_Semantic_Segmentation_CVPR_2020_paper.html) | 5/5/5 | Differentiable input-dependent gates select scale paths. | Include: generic routing prior art. |
| Sparse and Constrained Attention | 2018 | ACL | [ACL Anthology](https://aclanthology.org/P18-2059/) | 5/4/4 | Constrained softmax is a KL projection under simplex/upper bounds. | Include: exact optimization lineage. |
| FPR | 2023 | ICCV | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html) | 4/5/5 | False-positive CAM cues guide prototype-based background suppression. | Include: online-FP principle overlap. |
| Not All Pixels Are Equal | 2017 | CVPR | [CVF](https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Not_All_Pixels_CVPR_2017_paper.html) | 4/4/4 | Cascades easy regions early and routes harder regions deeper. | Omit from final 15: DEAL is closer to error-supervised attention. |
| Difficulty-Aware Attention with Confidence Learning | 2019 | AAAI | [AAAI/OJS](https://ojs.aaai.org/index.php/AAAI/article/view/3900) | 4/4/4 | Confidence learning and difficulty-aware attention emphasize hard regions. | Omit: useful background, lower mechanism match than DEAL/EAR-NET. |
| A Regularized Framework for Sparse and Structured Neural Attention | 2017 | NeurIPS | [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/2d1b2a5ff364606ff041650887723470-Abstract.html) | 5/4/4 | Differentiable structured/sparse attention through regularized max mappings. | Omit: ACL constrained softmax is closer to explicit KL bounds. |
| Sparse Sinkhorn Attention | 2020 | ICML | [PMLR](https://proceedings.mlr.press/v119/tay20a.html) | 4/5/5 | Differentiable sorting/balancing supports efficient sparse sequence attention. | Omit: efficiency/ordering goal differs from dual-risk projection. |
| Auto Learning Attention | 2020 | NeurIPS | [Proceedings](https://proceedings.neurips.cc/paper/2020/hash/103303dd56a731e377d01f6a37badae3-Abstract.html) | 4/3/3 | Searches heterogeneous attention operations in a DAG. | Omit: generic automatic design, no evidence roles or certificates. |
| HSTNet | 2025 | IEEE TGRS | [DOI](https://doi.org/10.1109/TGRS.2025.3608726) | 3/3/3 | Hybrid spatial-channel sparse transformer distinguishes target/background. | Omit: no inspected tri-role/error-supervised projection chain. |
| IDNANet | 2023 | arXiv preprint | [arXiv](https://arxiv.org/abs/2311.08747) | 3/3/3 | Transformer-enhanced DNANet and weighted segmentation loss. | Omit: lower source status and less direct than retained cross-level anchors. |
| DCCS-Det | 2026 | arXiv preprint | [arXiv](https://arxiv.org/abs/2601.16428) | 3/2/2 | Directional context and cross-scale saliency enhancement. | Omit: unverified venue and weaker match than CCDNet/NS-FPN. |

## Excluded Sources

- MDPI-domain search results were discarded immediately and were not cited, scored or used to support any claim.
- Secondary blogs, paper-summary sites, ResearchGate copies and search snippets were not used as final evidence.
- Low-signal or unverifiable preprints were excluded unless they were unusually close to the proposed mechanism; CCDNet is the sole retained preprint and is labeled accordingly.

## Evidence And Scoring Notes

- Scores are paper-quality judgments, not predicted acceptance probabilities and not evidence that the proposed model will outperform SCTransNet.
- `insight` measures the clarity/non-obviousness of the paper's central idea; `completeness` covers method, ablation, protocol and reproducibility; `numeric evidence` measures breadth/fairness of reported experiments.
- `Risk` means a work materially constrains novelty wording. It does not mean the work is higher quality than an `A` item.
- Comparisons between a retained paper and the proposed V3.2 contract are explicitly analytical inferences from the cited mechanism descriptions; the paper-specific descriptions themselves come from the linked primary source.

## Unknowns

- No inspected work combines all of the following in one published operator: leave-one-level-out cross-scale evidence, value-conditioned three-role spatial support, online outside-GT false-positive supervision, two simultaneous risk constraints, row-wise KL projection and emitted feasibility/KKT certificates. This is an absence in the screened set, not proof of worldwide priority.
- CCDNet was only verifiable as an arXiv preprint on the search date; later venue status is unknown here.
- The exact semantic meaning of “common” background and whether the three supports are identifiable without collapse remain experimental questions, not literature-established facts.
- Whether current-prediction-derived false-positive density improves generalization rather than amplifying early training errors is unknown and requires controlled ablation.
- Dataset aliases, hashes, train/validation/test splits, thresholding and connected-component rules must be matched before any baseline comparison. In particular, the ACM dataset source calls the dataset SIRST, while several codebases use the alias NUAA-SIRST.

## Handoff Notes

### For writing

- Cite SCTransNet when defining the host failure: SSCA performs global cross-level channel interaction, but do not assert that it necessarily propagates distractors without diagnostic evidence.
- Cite MDvsFA, NS-FPN and CCDNet when motivating the miss/false-alarm or distractor problem.
- Cite DEAL and EAR-NET when introducing error-derived router supervision; the defensible distinction is the tri-role support inside cross-scale attention, not “the first supervised error attention.”
- Cite Dynamic Routing when using the word “routing”; call the proposed operation **evidence-role routing within one SSCA attention map**, not generic dynamic routing.
- Cite Sparse and Constrained Attention for the KL/I-projection lineage; claim a new risk semantics/certificate implementation only if the mathematics and evidence support it.

### For idea optimization

- Keep one operator with necessary internal stages. Do not turn the literature gaps into several loosely connected modules.
- The strongest coherent contribution candidate is: **learned tri-role evidence supports that identify corroborating target, contradictory distractor and common background tokens, followed by an auditable dual-risk information projection inside SCTransNet SSCA**.
- Treat the training target construction and the projection/certificate as integral mechanisms of that operator, not as three independent “innovation points.”

### For experiment design

Minimum novelty-discriminating ablations:

1. SCTransNet host versus the complete operator under the same seed, split and schedule.
2. Static support versus learnable router.
3. Two roles (target/background) versus three roles (target/distractor/common background).
4. Q/K-only descriptors versus Q/K plus value content.
5. GT-only router targets versus GT plus detached current outside-GT false-positive density.
6. No projection, one-risk projection and two-risk projection.
7. Router without certificate emission versus identical outputs with certificates, to show certificates are observability rather than a hidden performance change.
8. Report mIoU, nIoU, Pd, Fa and F1; preserve separate `best_mIoU` and `best_Pd` checkpoints.

### For review

Likely reviewer attacks:

- “Error-supervised attention already exists” (DEAL/EAR-NET).
- “False-positive/distractor suppression already exists in IRSTD” (MDvsFA/NS-FPN/CCDNet).
- “Three branches/direct-versus-reverse attention already exist” (RAN).
- “Dynamic routing and constrained attention are established” (Dynamic Routing; constrained softmax).
- “The result comes from a new auxiliary loss rather than the proposed attention operator” (MSHNet makes this especially salient).

The response must be experimental separation, not stronger wording.

## Standard-Mode Checklist

- [x] Public, non-sensitive queries used.
- [x] MDPI and low-quality sources excluded.
- [x] Primary/stable sources prioritized.
- [x] Titles deduplicated.
- [x] 23 candidates screened; 15 retained.
- [x] Year, venue/source status, paper type, rationale and scores recorded.
- [x] Paper claims linked to sources; cross-paper conclusions marked as inference.
- [x] Closest-work clusters, remaining gaps and rescue routes recorded.
- [x] Benchmark/protocol cautions recorded.
- [x] Reusable Markdown and CSV artifacts written.
