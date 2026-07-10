export type Highlight = {
  title: string;
  /** Author list as published. Your name gets emphasized automatically. */
  authors: string;
  /** Venue + year, shown as the pill in the card corner. */
  meta: string;
  blurb: string;
  href: string;
};

// Titles, author lists, and DOIs verified against Crossref.
export const highlights: Highlight[] = [
  {
    title: 'Gumbel-Based Active Sparse Mobile Crowd Sensing with Time Series Transformer',
    authors: 'Victor Li, Carson Lam, Ting Li',
    meta: "IPCCC '25",
    blurb:
      'A learned Gumbel-noise sensor selection layer paired with a time series transformer, reducing reconstruction error on missing sensor data by up to 28%.',
    href: 'https://doi.org/10.1109/IPCCC66453.2025.11304689',
  },
  {
    title:
      'Ensemble Learning with Early Fusion of Kernel-Transformed and Classical Electrocardiogram Features for Chagas Disease Detection',
    authors: 'Victor M. Li, Runze Yan, Alex Fedorov, Jiaying Lu',
    meta: "CinC '25",
    blurb:
      'Ensemble framework over AutoGluon, ECG-FM, FFT, and wavelet features for 12-lead ECG classification. Placed 39th in the 2025 George B. Moody PhysioNet Challenge.',
    href: 'https://doi.org/10.22489/CinC.2025.080',
  },
  {
    title: 'Patched Forecasting with Gumbel-Based Selector for Sparse Mobile Crowd Sensing',
    authors: 'Victor Li, Carson Lam, Ting Li',
    meta: "IPCCC '25 · Poster",
    blurb:
      'Poster companion to the full paper, presented at IPCCC 2025 in Austin, TX.',
    href: 'https://doi.org/10.1109/IPCCC66453.2025.11304635',
  },
];
