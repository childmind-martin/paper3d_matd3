# APF Sanity Evidence Note

Appendix H.3 reports APF-only/APF-fusion evidence as a supplementary sanity/reference check. It answers a design question that is different from the main semantic-ablation question:

> learned action alone, APF alone, or learned-action + APF correction?

This check helps justify retaining APF correction as part of the execution interface. It indicates whether learned motion proposals and APF correction provide complementary signals in the controlled fixed-layout APF study.

## Relation to the Main Claim

The APF-only/APF-fusion study is orthogonal to the main dual-semantic ablation. The main paper claim asks whether, after APF-corrected execution is introduced, the learning system should preserve raw/corrected action roles in replay, critic evaluation, target-side construction, and actor-gradient routing.

That main question is tested by Table 5, not by Appendix H.3.

## Evidence Strength

Appendix H.3 is supplementary single-run or weaker-statistics evidence. It should not be presented as equal to Table 5 or Table 6:

- Table 5 is the core semantic-ablation mechanism evidence.
- Table 6 is the primary Level-2 checkpoint-FR deployment evidence.
- Appendix H.3 is retained for transparency and sanity/reference validation.

## Interpretation Boundary

Appendix H.3 should not be used to claim that APF-only methods generally fail, to make broad superiority claims for APF-fusion, or to establish that the dual-semantic mechanism is necessary. It is not used to establish the necessity of dual-semantic replay, critic construction, target reconstruction, or separated-gradient routing.

Its role is narrower: it documents why the paper keeps learned-action + APF correction in the execution interface before studying how that corrected execution should be represented inside actor--critic learning.
