export type Highlight = {
    title: string;
    /** Author list as published. Your name gets emphasized automatically. */
    authors: string;
    /** Venue + year, shown as the pill in the card corner. */
    meta: string;
    blurb: string;
    /** Omit for unpublished work — the card renders as plain text, not a link. */
    href?: string;
};

// Under review first, then published newest-first.
// Titles, author lists, and DOIs verified against Crossref.
export const highlights: Highlight[] = [
    {
        title: "Reconciling Set-Valued Policy & Dead-End Discovery in Healthcare Reinforcement Learning: An Empirical Analysis",
        authors: "Victor Li, Sixing Wu, Shengpu Tang",
        meta: "Under review",
        blurb: "An empirical study of how consistently Set-Valued Policies and Dead-End Discovery agree on clinician-in-the-loop sepsis treatment, plus a partial ordering over recommended actions. Evaluated on LifeGate and MIMIC-III.",
    },
    {
        title: "CCQ: An LLM-Curated Child Care Quality Dataset to Support AI Research for Children’s Health",
        authors:
            "Victor Li, Yuzhang Xie, Qingyang Zhu, Wenjing Ma, Xiao Hu, Carl Yang, Jinbing Bai, Huiwen Xu, Jiaying Lu",
        meta: "Under review",
        blurb: "An LLM-curated benchmark of child care quality and provider compliance ratings across 12 U.S. states, with within-state and leave-one-state-out transfer tasks over language models, LLMs, and tabular baselines.",
    },
    {
        title: "Gumbel-Based Active Sparse Mobile Crowd Sensing with Time Series Transformer",
        authors: "Victor Li, Carson Lam, Ting Li",
        meta: "IPCCC '25",
        blurb: "A learned Gumbel-noise sensor selection layer paired with a time series transformer, reducing reconstruction error on missing sensor data by up to 28% on Urban Air and SensorScope St-Bernard datasets.",
        href: "https://doi.org/10.1109/IPCCC66453.2025.11304689",
    },
    {
        title: "Patched Forecasting with Gumbel-Based Selector for Sparse Mobile Crowd Sensing",
        authors: "Victor Li, Carson Lam, Ting Li",
        meta: "IPCCC '25",
        blurb: "A follow-up extended abstract that convolutionally splits each sensing cycle into patches, each with its own selector layer, testing whether intra-cycle structure improves sensor selection.",
        href: "https://doi.org/10.1109/IPCCC66453.2025.11304635",
    },
    {
        title: "Ensemble Learning with Early Fusion of Kernel-Transformed and Classical Electrocardiogram Features for Chagas Disease Detection",
        authors: "Victor M. Li, Runze Yan, Alex Fedorov, Jiaying Lu",
        meta: "CinC '25",
        blurb: "Ensemble framework over AutoGluon, ECG-FM, FFT, and wavelet features for 12-lead ECG classification. Placed 39th in the 2025 George B. Moody PhysioNet Challenge.",
        href: "https://doi.org/10.22489/CinC.2025.080",
    },
    {
        title: "Softening Overly Demanding Requirements in Recommendation System",
        authors: "Haoyu Hu, Jinyi Guo, Victor Li, Yuzhang Li",
        meta: "ISCAIS '23",
        blurb: "Cuts item under-recommendation bias and training cost in Debiased Bayesian Personalized Ranking by replacing the adversarial debiasing network with an autoencoder plus ranking post-processing.",
        href: "https://doi.org/10.1117/12.2683667",
    },
];
