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
        date: "Sep 2026",
        title: "SVP-DeD paper accepted to PSB '27",
        text: "“Reconciling Set-Valued Policy & Dead-End Discovery in Healthcare Reinforcement Learning: An Empirical Analysis” was accepted to the Pacific Symposium on Biocomputing 2027.",
    },
    {
        date: "Apr 2026 —",
        title: "Type 1 diabetes digital-twin benchmark",
        text: "A benchmark that scores Type 1 diabetes digital twins by decision quality — whether a twin ranks candidate insulin treatments the way the real patient would — rather than by trajectory error alone. The two criteria disagree: the worst-RMSE twin ranked treatments second best, while a twin with half the RMSE ranked at chance.",
    },
    {
        date: "Oct 2025 —",
        title: "Child care quality dataset",
        text: "CCQ, a de-identified dataset of 64,479 child care providers across 12 U.S. states, curated from state QRIS portals by an LLM pipeline and released on Hugging Face. Benchmarks show tree models win within-state, while zero-shot transfer to a new state lands near chance.",
    },
];
