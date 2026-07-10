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
    date: 'In progress',
    title: 'Type 1 diabetes benchmark',
    // TODO: I don't have details on this one — it isn't on your resume.
    // Replace with what the benchmark actually measures and what it's built on.
    text: 'Building a benchmark for machine learning on type 1 diabetes data.',
  },
  {
    date: 'Oct 2025 —',
    title: 'Child care quality dataset',
    text: 'A new AI/ML benchmark dataset of child care quality ratings, scraped from the Georgia Department of Early Care and Learning. Currently building a causal discovery graph via FPM and PC, with a DiCE counterfactual framework to rank the key determinants of quality across 245 provider attributes.',
  },
];
