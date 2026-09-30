export type Highlight = {
    title: string;
    /** Author list as published. Your name gets emphasized automatically. */
    authors: string;
    /** Venue + year, shown as the pill in the card corner. */
    meta: string;
    /**
     * Drives the pill style and CV ordering. Anything not "published" gets a
     * muted pill so it doesn't read as a venue.
     */
    status: "published" | "preprint" | "review";
    blurb: string;
    /** Omit when there's nothing to link yet — the card renders as plain text. */
    href?: string;
    /** Set false to list a paper on the CV only, not as a home-page highlight. */
    showOnHome?: boolean;
};

// Preprints and under review first, then published newest-first.
// Titles, author lists, and DOIs verified against Crossref.
export const highlights: Highlight[] = [
    {
        title: "Evaluating Digital Twins for Type 1 Diabetes by Decision Quality",
        authors:
            "Victor Li, Owen Tucker, Michael S. Hughes, Temiloluwa Prioleau, Shengpu Tang",
        meta: "Preprint",
        status: "preprint",
        blurb: "An evaluation protocol that scores a Type 1 diabetes digital twin by whether its glucose trajectories rank candidate insulin treatments as the real patient would, not by trajectory error alone. On 30 UVA/Padova virtual patients under 21 treatments, the two criteria disagree: the worst-RMSE twin ranks second best, and a twin with half the RMSE ranks at chance.",
    },
    {
        title: "CCQ: A Multi-State Child Care Quality Dataset to Support AI for Children’s Health Research",
        authors:
            "Victor Li, Yuzhang Xie, Ziwei Dong, Wenjing Ma, Carl Yang, Jinbing Bai, Huiwen Xu, Jiaying Lu",
        meta: "Preprint",
        status: "preprint",
        blurb: "A de-identified dataset of 64,479 child care providers across 12 U.S. states, curated from state QRIS portals by an LLM pipeline and released on Hugging Face, with within-state and leave-one-state-out benchmarks over tabular models and language models.",
    },
    {
        title: "Physiological Identifiability of Type 1 Diabetes Digital Twins Under Behavioral Heterogeneity",
        authors:
            "Owen Tucker, Victor Li, Michael S. Hughes, Temiloluwa Prioleau, Shengpu Tang",
        meta: "Under review",
        status: "review",
        blurb: "A study of when the physiological parameters of a Type 1 diabetes digital twin can actually be recovered from observed data, and how behavioral variation across patients limits that identifiability.",
    },
    {
        title: "Hierarchical Timeline Generation and Visualization for Child Care Quality Monitoring Reports",
        authors:
            "Aria Pan, Victor Li, Ziwei Dong, Yuzhang Xie, Huiwen Xu, Carl Yang, Jiaying Lu",
        meta: "Under review",
        status: "review",
        blurb: "Generating and visualizing hierarchical timelines from child care quality monitoring reports.",
        // Still a work in progress — CV only until it's ready to feature.
        showOnHome: false,
    },
    {
        title: "Reconciling Set-Valued Policy & Dead-End Discovery in Healthcare Reinforcement Learning: An Empirical Analysis",
        authors: "Victor Li, Sixing Wu, Shengpu Tang",
        meta: "PSB '27",
        status: "published",
        // Proceedings aren't out yet — add the DOI as href once they are.
        blurb: "An empirical study of how consistently Set-Valued Policies and Dead-End Discovery agree on clinician-in-the-loop sepsis treatment, plus a partial ordering over recommended actions. Evaluated on LifeGate and MIMIC-III.",
    },
    {
        title: "Gumbel-Based Active Sparse Mobile Crowd Sensing with Time Series Transformer",
        authors: "Victor Li, Carson Lam, Ting Li",
        meta: "IPCCC '25",
        blurb: "A learned Gumbel-noise sensor selection layer paired with a time series transformer, reducing reconstruction error on missing sensor data by up to 28% on Urban Air and SensorScope St-Bernard datasets.",
        status: "published",
        href: "https://doi.org/10.1109/IPCCC66453.2025.11304689",
    },
    {
        title: "Patched Forecasting with Gumbel-Based Selector for Sparse Mobile Crowd Sensing",
        authors: "Victor Li, Carson Lam, Ting Li",
        meta: "IPCCC '25",
        blurb: "A follow-up extended abstract that convolutionally splits each sensing cycle into patches, each with its own selector layer, testing whether intra-cycle structure improves sensor selection.",
        status: "published",
        href: "https://doi.org/10.1109/IPCCC66453.2025.11304635",
    },
    {
        title: "Ensemble Learning with Early Fusion of Kernel-Transformed and Classical Electrocardiogram Features for Chagas Disease Detection",
        authors: "Victor M. Li, Runze Yan, Alex Fedorov, Jiaying Lu",
        meta: "CinC '25",
        blurb: "Ensemble framework over AutoGluon, ECG-FM, FFT, and wavelet features for 12-lead ECG classification. Placed 39th in the 2025 George B. Moody PhysioNet Challenge.",
        status: "published",
        href: "https://doi.org/10.22489/CinC.2025.080",
    },
    {
        title: "Softening Overly Demanding Requirements in Recommendation System",
        authors: "Haoyu Hu, Jinyi Guo, Victor Li, Yuzhang Li",
        meta: "ISCAIS '23",
        blurb: "Cuts item under-recommendation bias and training cost in Debiased Bayesian Personalized Ranking by replacing the adversarial debiasing network with an autoencoder plus ranking post-processing.",
        status: "published",
        href: "https://doi.org/10.1117/12.2683667",
    },
];
