# Cover Letter — Entertainment Computing

Lee ChangHo (이창호)
Independent Researcher, Republic of Korea
damelazydeveloper@gmail.com

2026-07-03

---

Dear Editors of *Entertainment Computing*,

I am pleased to submit my manuscript, **"An Explainable Rule-Based Approach to
Automatic Rhythm Game Chart Generation,"** for consideration as a research
article in *Entertainment Computing*.

**Scope fit.** Chart creation — authoring the note sequences that players hit —
is the core content-creation task of rhythm-based music games such as DJMAX and
Project Sekai, and it directly shapes the player experience. The manuscript
frames automatic chart generation as a *procedural content generation (PCG)*
problem for music games and presents a complete, working system that turns
arbitrary audio (a local file or a YouTube link) into playable 4–8 key charts.
As such, the work sits squarely within the journal's interest in game content,
procedural generation, and interactive entertainment technology.

**Contribution.** Rather than a learned black box, the proposed system encodes
musical and charting design principles as explicit rules, and combines: (i) a
lightweight, pure-NumPy music-information-retrieval pipeline (BPM/beat/tempo map,
band separation, section structure, and seven musical feature vectors); (ii) a
feature-based pattern-scoring scheduler with contour-based motif memory and a
closed-loop density controller; and (iii) a six-dimensional quality evaluator
with multi-seed selection. The system is reproducible under a fixed random seed
while allowing controlled probabilistic variation, and it records post-hoc,
traceable rationale for every note. It runs on a laptop-class CPU with no GPU,
deep-learning framework, or training data, processing a 3.5-minute song in about
1.4 seconds.

**Scope of claims.** I am careful to scope the empirical claims: the paper
demonstrates *conformance to predefined charting rules*, not the reproduction of
human-perceived quality, which is identified explicitly as future work (a user
study and a controlled comparison against learning-based generators). I believe
this transparent framing strengthens rather than weakens the contribution.

**Originality and ethics.** The system does not learn from or copy any
individual charter's work; it codifies design principles that are commonly
perceived as "good charting" across several commercial games. The full source
code, sample data, and both Korean and English versions of the paper are openly
available under the MIT license (repository URL provided in the non-anonymized
title page and withheld from the blind manuscript for review), and all results
are reproducible with the procedure in the appendix.

This manuscript is original, has not been published elsewhere, and is not under
consideration by any other journal. There are no conflicts of interest to
declare. As an independent researcher, I am the sole author.

Thank you for your time and consideration. I look forward to the reviewers'
feedback.

Sincerely,

Lee ChangHo (이창호)
Independent Researcher, Republic of Korea
damelazydeveloper@gmail.com
