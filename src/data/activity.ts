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
        text: "A benchmark that scores digital twins of Type 1 diabetes patients by decision quality: whether a fitted twin ranks candidate insulin treatments the way the real patient would, rather than by trajectory error alone. Built on 30 UVA/Padova virtual patients under 21 basal-bolus treatments, comparing MCMC, simulation-based inference, and Kalman filter twins against linear, neural, and population-prior baselines. The two criteria disagree: the worst-RMSE twin ranked treatments second best, while a twin with half the RMSE ranked at chance. Two papers submitted to the Sim2Science workshop at NeurIPS 2026.",
    },
    {
        date: "Oct 2025 —",
        title: "Child care quality dataset",
        text: "CCQ, a de-identified dataset of 64,479 child care providers across 12 U.S. states, collected from state QRIS portals and released on Hugging Face as row-aligned text and preprocessed tabular versions. An LLM curation pipeline adapts a hand-built Georgia reference to anonymize and clean the remaining states with open-source Qwen3 agents, and within-state and leave-one-state-out benchmarks pit tabular models against language models: trees win within-state, zero-shot transfer lands near chance, and modest target-state supervision recovers most of the gap.",
    },
];
