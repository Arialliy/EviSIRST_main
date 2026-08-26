# Literature Search: C³-SBSC V3.2 Learnable Tri-Role Evidence Routing

Date: 2026-08-25  
Mode: standard  
Search purpose: establish the nearest prior art, claim boundary and evidence requirements before freezing a SCTransNet-L1 operator that learns consistent-target, contradictory-distractor and common-background spatial supports, then applies a dual-risk row-wise KL/I-projection with numerical certificates.  
Target venue/family: strong computer-vision / remote-sensing venue; no venue-specific formatting decision yet.  
Source-quality policy: public primary sources only; MDPI excluded.

## Summary

- **Closest host work:** SCTransNet is the exact baseline and already claims full-scale target/clutter semantic reinforcement through SCTB/SSCA. The new work cannot claim “cross-scale target-background modeling” as its novelty.
- **Closest problem work:** MDvsFA already decomposes miss detection and false alarm; NS-FPN directly targets false alarms through noise suppression; CCDNet explicitly learns to discriminate target-like distractors. “Solving false alarms/distractors” is therefore a motivation, not a novel contribution.
- **Closest supervision work:** DEAL supervises a pixel-wise difficulty/attention branch from current prediction errors; EAR-NET supervises error attention with prediction-versus-GT differences; FPR learns from false-positive cues. A broad “first error-supervised attention” or “first online false-positive guidance” claim is indefensible.
- **Closest routing/role work:** Dynamic Routing learns input-dependent scale paths, while Reverse Attention Network learns direct, reverse and interaction branches. The proposed router must be described narrowly as **within-attention evidence-role allocation**, not the invention of routing or opposite-role attention.
- **Closest optimization work:** constrained softmax already performs KL projection of softmax attention onto a constrained simplex. The potentially defensible part is the IRSTD-specific pair of risk semantics, exact per-row feasibility/KKT handling and evidence certificates—not KL-constrained attention in general.
- **Conditional novelty boundary:** the screened set did not reveal one method that joins (i) leave-one-level-out cross-scale evidence, (ii) value-conditioned three-role spatial support trained with target/online false-positive/common-background targets, and (iii) auditable two-risk row-wise information projection inside SCTransNet SSCA. This is a search finding, not a priority proof.
- **Publication consequence:** the idea remains viable only if ablations show that all three roles are identifiable and useful, that online false-positive supervision is not the sole source of gains, and that the dual-risk projection improves the Pd/Fa trade-off without harming mIoU/nIoU/F1.

## Paper Table

Scores use 1–5 anchors from the CCFA literature-search protocol. `Risk` denotes a novelty-sensitive work, not a quality rank.

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Evidence and relevance |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection | 2024 | IEEE TGRS | [DOI](https://doi.org/10.1109/TGRS.2024.3383649) | pure method | 4 | 4 | 4 | Risk | Exact host baseline. SCTB/SSCA interacts all encoder outputs and redistributes mixed features to decoders to reinforce target/clutter differences. |
| 2 | Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images | 2019 | ICCV | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.html) | pure method | 5 | 4 | 4 | Risk | Explicitly frames MD and FA as opposing subproblems and assigns them to adversarially trained models. |
| 3 | Asymmetric Contextual Modulation for Infrared Small Target Detection | 2021 | WACV | [CVF](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html) | method + benchmark | 4 | 4 | 4 | A | Combines top-down global context and bottom-up point-wise channel modulation; introduces the SIRST benchmark. |
| 4 | Attentional Local Contrast Networks for Infrared Small Target Detection | 2021 | IEEE TGRS | [DOI](https://doi.org/10.1109/TGRS.2020.3044958) | pure method | 4 | 4 | 4 | A | Embeds a parameter-free local-contrast operator and bottom-up attentional modulation, limiting novelty claims based only on static rarity/contrast. |
| 5 | Dense Nested Attention Network for Infrared Small Target Detection | 2023 | IEEE TIP | [DOI](https://doi.org/10.1109/TIP.2022.3199107) | method + benchmark | 4 | 5 | 5 | A | Dense cross-level interaction plus channel/spatial attention; introduces NUDT-SIRST and formalizes target-level Pd/Fa alongside IoU. |
| 6 | ISNet: Shape Matters for Infrared Small Target Detection | 2022 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html) | method + benchmark | 4 | 5 | 5 | A | Edge/shape-aware alternative and source of IRSTD-1K, the benchmark on which clutter separation is especially important. |
| 7 | Infrared Small Target Detection with Scale and Location Sensitivity | 2024 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | pure method | 4 | 5 | 5 | Risk | Shows that a purpose-built loss plus a simple multi-scale head can drive performance; router supervision must be isolated from operator effects. |
| 8 | Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective | 2026 | CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | pure method | 4 | 4 | 4 | Risk | Recent direct low-false-alarm work; uses frequency-guided purification and spiral-aware fusion rather than constrained cross-scale attention. |
| 9 | CCDNet: Learning to Detect Camouflage against Distractors in Infrared Small Target Detection | 2026 | arXiv preprint | [arXiv](https://arxiv.org/abs/2603.29228) | pure method | 4 | 3 | 3 | Risk | Closest distractor language: local/global contrastive similarity is used to discriminate target-like backgrounds. Venue status was not verified beyond preprint. |
| 10 | DEAL: Difficulty-aware Active Learning for Semantic Segmentation | 2020 | ACCV | [CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Xie_DEAL_Difficulty-aware_Active_Learning_for_Semantic_Segmentation_ACCV_2020_paper.html) | pure method | 4 | 4 | 4 | Risk | A pixel-wise attention/difficulty branch is supervised by the error between segmentation output and GT. |
| 11 | EAR-NET: Error Attention Refining Network for Retinal Vessel Segmentation | 2021 | DICTA | [DOI](https://doi.org/10.1109/DICTA52665.2021.9647299) | pure method | 4 | 3 | 3 | Risk | Treats initial prediction-versus-GT differences as supervision for an error-attention map used in refinement. |
| 12 | Semantic Segmentation with Reverse Attention | 2017 | BMVC | [BMVA](https://bmva-archive.org.uk/bmvc/2017/papers/paper018/paper018.pdf) | pure method | 4 | 4 | 4 | Risk | Learns direct, reverse and reverse-attention branches, establishing prior art for opposite concepts and three-way branch structure. |
| 13 | Learning Dynamic Routing for Semantic Segmentation | 2020 | CVPR | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Learning_Dynamic_Routing_for_Semantic_Segmentation_CVPR_2020_paper.html) | pure method | 5 | 5 | 5 | Risk | Learns input-dependent soft gates over scale-transform paths; generic “dynamic routing” is already established. |
| 14 | Sparse and Constrained Attention for Neural Machine Translation | 2018 | ACL | [ACL Anthology](https://aclanthology.org/P18-2059/) | pure method | 5 | 4 | 4 | Risk | Defines constrained softmax as the KL-nearest distribution to softmax under simplex and upper-bound constraints. |
| 15 | FPR: False Positive Rectification for Weakly Supervised Semantic Segmentation | 2023 | ICCV | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_FPR_False_Positive_Rectification_for_Weakly_Supervised_Semantic_Segmentation_ICCV_2023_paper.html) | pure method | 4 | 5 | 5 | Risk | Mines false-positive CAM cues and learns positive/negative prototypes and rectification losses to suppress co-occurring background. |

## Closest-Work Ranking For The Proposed Contract

| Rank | Work | Exact overlap | Material difference | Risk to claim |
| ---: | --- | --- | --- | --- |
| 1 | SCTransNet | Same host, same SCTB/SSCA cross-level attention location. | No learned tri-role support or audited dual-risk projection in the published operator. | Any benefit must be attributed to a precise SSCA repair under an otherwise unchanged host. |
| 2 | DEAL | Current segmentation error plus GT supervises an attention/difficulty branch. | Active-learning acquisition goal; binary difficulty rather than three evidence roles inside cross-scale attention. | “First prediction-error-supervised attention” is not defensible. |
| 3 | MDvsFA | Explicitly separates the opposing demands of detection and false-alarm suppression in IR imagery. | Uses adversarial multi-model decomposition, not within-row attention allocation. | “First to disentangle detection and false alarm” is not defensible. |
| 4 | Sparse and Constrained Attention | KL projection of softmax attention under explicit constraints. | NMT coverage bounds rather than two IRSTD evidence-risk constraints and row certificates. | “Novel KL-constrained attention/I-projection” is not defensible. |
| 5 | EAR-NET | Error-map supervision teaches an attention map that refines segmentation errors. | Two-stage retinal pipeline, not one SSCA operator or online three-role distribution. | Router-label construction requires explicit differentiation and ablation. |
| 6 | CCDNet | Explicit target-versus-distractor learning to reduce IRSTD false alarms. | Contrastive discriminator/neck rather than evidence-role attention and feasibility projection. | Avoid first distractor-aware IRSTD claims; preprint status should be stated. |
| 7 | Reverse Attention Network | Direct/reverse/interaction three-branch learning. | Semantic class/opposite branches rather than three independently normalized, potentially overlapping spatial supports tied to cross-level evidence and risks. | Do not market “three roles” alone as the innovation. |
| 8 | Dynamic Routing | Learnable, data-dependent routing across scales. | Selects computation paths; proposed router allocates attention support roles within a fixed SSCA path. | Qualify the routing object and granularity every time. |
| 9 | FPR | False-positive cues guide suppression of confusing background. | Weakly supervised CAM/prototype setting rather than fully supervised IRSTD cross-level attention. | Avoid first online-FP-mining claims. |
| 10 | NS-FPN | Directly targets increased false alarms in IRSTD. | Frequency/noise suppression route, not learned evidence roles. | Establish comparative baseline or at least discuss it as a distinct recent solution. |

## Clusters

### Cluster 1: IRSTD false alarms, clutter and distractors

- **Representative papers:** MDvsFA, ALCNet, NS-FPN, CCDNet.
- **What is already solved:** miss/false-alarm decomposition, local-contrast priors, frequency-domain noise suppression, and learned target-versus-distractor discrimination have all been proposed.
- **Remaining gap:** these works do not, in the inspected sources, formulate the SCTransNet attention row as an auditable allocation problem among corroborating target, contradictory distractor and common background evidence.
- **Possible differentiation:** demonstrate that the router discovers different supports for true targets and target-like false objects, and that the projection improves the joint Pd/Fa frontier rather than merely reducing activation magnitude.
- **How it affects this paper:** false-alarm reduction is an outcome claim requiring measured Pd/Fa; it is not an innovation label.

### Cluster 2: Cross-scale context and small-target preservation

- **Representative papers:** SCTransNet, ACM, DNANet, ALCNet, ISNet.
- **What is already solved:** multi-level context exchange, bottom-up/top-down modulation, dense nested fusion, attention enhancement and shape/edge preservation.
- **Remaining gap:** cross-scale fusion normally decides how to aggregate useful information, but the screened host literature does not expose a row-wise, certificate-bearing decision about which cross-level positions are consistent, contradictory or common.
- **Possible differentiation:** hold the host architecture fixed and replace only L1 SSCA behavior; show causal ablations for leave-one-level-out evidence and value content.
- **How it affects this paper:** “better multi-scale fusion” is too generic. The claim must be about evidence-role-conditioned constrained allocation.

### Cluster 3: Error, difficulty and false-positive supervision

- **Representative papers:** DEAL, EAR-NET, FPR, MSHNet, Reverse Attention Network.
- **What is already solved:** hard-region routing, error-supervised attention, false-positive rectification, direct/reverse semantic branches and purpose-designed auxiliary/loss signals.
- **Remaining gap:** the inspected work does not use a normalized three-role target in which GT target mass, detached outside-GT prediction mass and residual common background jointly supervise one cross-scale attention router.
- **Possible differentiation:** use continuous, threshold-free role targets; publish role-mass/collapse diagnostics; compare binary versus tri-role and GT-only versus GT+online-FP targets.
- **How it affects this paper:** the auxiliary objective is part of the method and must have its own ablation; it cannot be hidden as implementation detail.

### Cluster 4: Dynamic routing and learned attention structure

- **Representative paper:** Learning Dynamic Routing for Semantic Segmentation.
- **What is already solved:** differentiable input-conditioned routing with soft gates over scale paths and optional compute constraints.
- **Remaining gap:** routing is not tied to evidence roles or attention-row feasibility.
- **Possible differentiation:** define the routed object mathematically: three spatial support distributions inside a single fixed SSCA operator, not architecture/path selection.
- **How it affects this paper:** use a precise modifier such as “tri-role evidence router”; never claim generic dynamic routing.

### Cluster 5: Constrained and information-projected attention

- **Representative paper:** Sparse and Constrained Attention for Neural Machine Translation.
- **Supporting near miss:** A Regularized Framework for Sparse and Structured Neural Attention (screened but omitted from the final 15).
- **What is already solved:** differentiable constrained attention, simplex projection, sparsity/structure and KL-nearest constrained softmax.
- **Remaining gap:** the inspected source constrains coverage/fertility, not two learned IRSTD evidence risks with explicit feasibility, numerical fallbacks and per-row KKT certificates.
- **Possible differentiation:** claim the domain-specific risk construction and auditable solver contract, then verify it with public test fixtures and intervention diagnostics.
- **How it affects this paper:** the projection mathematics requires lineage citation; novelty, if any, lies in the risk semantics and closed-loop operator.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| IRSTD false alarms/distractors | crowded but open | No inspected SCTransNet-row formulation separates corroboration, contradiction and common background with feasibility guarantees. | Tri-role within-SSCA evidence allocation. | Full-validation role separation; Pd/Fa curves; connected-component FP analysis; comparison to recent false-alarm baselines. | High wording overlap with MDvsFA, NS-FPN and CCDNet. |
| Cross-scale attention | crowded but open | Existing fusion is not explicitly contradiction-aware or certificate-bearing. | Leave-one-level-out peer evidence plus value content in one L1 replacement. | Host-isolation tests; Q/K versus Q/K/V ablation; level intervention studies. | Generic fusion/attention claim will be rejected as incremental. |
| Error-supervised routing | covered central claim | Prediction errors supervising attention/difficulty are established. | Three independently normalized, potentially overlapping continuous roles tied to attention support rather than a generic error mask. | Binary-vs-three-role, GT-only-vs-online-FP, detach/no-detach and collapse diagnostics. | Very high due DEAL/EAR-NET/FPR. |
| Dynamic routing | covered central claim | Data-dependent routing itself is established. | Route evidence roles rather than network paths. | Fixed-path proof, parameter/FLOP audit, input-conditioned role maps. | High terminology risk. |
| KL-constrained attention | covered central claim | KL-nearest constrained softmax is established. | Two IRSTD evidence risks plus exact row certificates. | Feasibility/KKT tests, fallback rate, AMP consistency and projection ablations. | High mathematical-prior-art risk. |
| Benchmark protocol | benchmark gap | Split aliases and threshold/component conventions can invalidate small metric gains. | Freeze hashes, split, resize, threshold and component matching before comparison. | Reproducibility manifest and same-protocol baseline rerun. | High empirical-comparability risk. |

## Benchmark And Dataset Candidates

| Name | Source | Task | Metrics to preserve | Baselines | Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| SIRST (often aliased NUAA-SIRST in codebases) | [ACM/SIRST source](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html) | Single-frame IR small-target segmentation | mIoU, nIoU, Pd, Fa, F1 | ACM, ALCNet, DNANet, ISNet, SCTransNet, MSHNet | Required by the project | Alias/version/split mismatch; record file hashes and exact partition. |
| NUDT-SIRST | [DNANet](https://doi.org/10.1109/TIP.2022.3199107) | Synthetic/multi-scene SIRST segmentation and detection | mIoU, nIoU, Pd, Fa, F1 | DNANet and SCTransNet | Required by the project | Synthetic composition and target-level matching rules must match. |
| IRSTD-1K | [ISNet](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html) | Realistic, clutter-rich pixel-level IRSTD | mIoU, nIoU, Pd, Fa, F1 | ISNet, DNANet, MSHNet, SCTransNet, NS-FPN | Primary stress test for target-like clutter | High variance from split, resize and post-processing; do not tune on official test. |

## Defensible And Indefensible Claims

| Claim form | Decision | Reason / condition |
| --- | --- | --- |
| “We are the first to address false alarms in IRSTD.” | Indefensible | MDvsFA and NS-FPN directly address false alarms; CCDNet addresses distractors. |
| “We introduce prediction-error-supervised attention.” | Indefensible | DEAL and EAR-NET are direct precedents. |
| “We introduce dynamic routing for segmentation.” | Indefensible | Dynamic Routing (CVPR 2020) is direct prior art. |
| “We introduce KL-constrained attention.” | Indefensible | Constrained softmax (ACL 2018) already uses KL projection under attention constraints. |
| “We introduce a three-branch direct/reverse attention design.” | Indefensible | Reverse Attention Network already learns three related branches. |
| “We introduce an evidence-role router inside SCTransNet SSCA.” | Conditionally defensible | Must define the routed object and distinguish it empirically from generic gates, reverse attention and error maps. |
| “Three learned supports represent corroborating target, contradictory distractor and common background evidence.” | Conditionally defensible | Requires role-identifiability, non-collapse and intervention evidence on frozen validation data. |
| “Dual-risk information projection yields auditable attention allocation.” | Conditionally defensible | Must expose feasibility/KKT/fallback certificates and show the claimed risks differ from coverage bounds in prior constrained attention. |
| “The complete model stably exceeds SCTransNet.” | Unknown | Requires real same-protocol results on all three datasets and all five metrics; no literature search can establish this. |

## Required Novelty-Discriminating Evidence

1. **Host isolation:** SCTransNet and the complete method differ only at the declared L1 SSCA replacement plus its training objective.
2. **Role necessity:** static support, binary target/background routing and full tri-role routing under an identical schedule.
3. **Content necessity:** Q/K descriptors versus Q/K plus value content.
4. **Supervision necessity:** GT-only versus GT plus detached current outside-GT false-positive density; report early-epoch stability and role collapse.
5. **Projection necessity:** unconstrained router, one-risk projection and dual-risk projection; report mIoU/nIoU/Pd/Fa/F1 rather than mIoU alone.
6. **Closed-loop validity:** full-validation semantic lower bounds, numerical solver tests, AMP consistency, intervention effects and fallback rates before long training.
7. **Checkpoint protocol:** preserve physical `best_mIoU` and `best_Pd` weights because the novelty explicitly concerns the detection/false-alarm trade-off.

## Citation And Positioning Cautions

- The exact host and benchmark line should cite SCTransNet, ACM/SIRST, DNANet/NUDT-SIRST and ISNet/IRSTD-1K.
- The false-alarm motivation needs MDvsFA and recent NS-FPN; CCDNet should be labeled as a preprint if discussed.
- The supervision lineage needs DEAL, EAR-NET and FPR.
- “Routing” needs the Dynamic Routing citation and a narrow definition.
- The projection section needs the ACL constrained-attention citation; do not imply the KL/I-projection operation itself is new.
- MSHNet makes a likely reviewer concern concrete: any gain from router supervision must be separated from the structural projection mechanism.
- No result, superiority or stability claim is supported by this search. Those claims remain gated on actual same-protocol experiments.
