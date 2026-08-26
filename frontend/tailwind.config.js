/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#0f172a",
          soft: "#475569",
          faint: "#94a3b8",
        },
        accent: {
          DEFAULT: "#4f46e5",
          soft: "#eef2ff",
          ring: "#c7d2fe",
        },
        canvas: "#f8fafc",
        card: "#ffffff",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.12)",
      },
    },
  },
  plugins: [],
};
