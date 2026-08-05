export type Activity = {
    /** Any string. Displayed verbatim, so "Jul 2026" or "Summer 2026" both work. */
    date: string;
    title: string;
    text: string;
    href?: string;
};

// Newest first. Keep the last ~8; older entries belong on the CV.
export const activity: Activity[] = [
    {
        date: "Apr 2026 —",
        title: "Type 1 diabetes digital-twin benchmark",
        text: "A benchmark that scores digital twins of Type 1 diabetes patients by decision transfer: whether a fitted twin ranks candidate insulin therapies the way the real patient would, rather than by trajectory fidelity alone. Comparing per-instance Bayesian (MCMC) and simulation-based inference twins against neural-regressor baselines, with the UVA/Padova simulator as ground truth.",
    },
    {
        date: "Oct 2025 —",
        title: "Child care quality dataset",
        text: "A new LLM-curated AI/ML benchmark dataset of child care quality and provider compliance ratings spanning 12 U.S. states, with a within-state and leave-one-state-out transfer benchmark over pretrained language models, LLMs, and tabular baselines. Also building a causal discovery graph via FPM and PC, with a DiCE counterfactual framework to rank the key determinants of quality across 245 provider attributes on the Georgia subset.",
    },
];
