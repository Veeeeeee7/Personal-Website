// ─────────────────────────────────────────────────────────────
// Everything you'll want to edit regularly lives in src/data/.
// Nothing here requires touching a component.
// ─────────────────────────────────────────────────────────────

export const site = {
    name: "Victor Li",
    url: "https://victor-li.me",
    email: "hello@victor-li.me",

    github: "https://github.com/Veeeeeee7",
    linkedin: "https://www.linkedin.com/in/victor-li-85b56027b",

    description:
        "Victor Li — undergraduate researcher at Emory working on machine learning for healthcare, sparse sensing, and causal inference.",
} as const;

// Profile photo. Both files live in public/.
export const avatar = {
    webp: "/avatar.webp",
    jpg: "/avatar.jpg",
    alt: "Victor Li",
} as const;

// The "About" block on the home page — bare facts, one line each.
export const facts: string[] = [
    "B.S. Mathematics and Computer Science",
    "Emory University, Class of 2027",
    "Atlanta, Georgia",
];
