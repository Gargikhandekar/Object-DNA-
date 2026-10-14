/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: { 950: "#0b0d10", 900: "#111418", 850: "#161a1f", 800: "#1c2128", 700: "#272e37", 600: "#384250", 500: "#5b6675" },
        accent: { DEFAULT: "#22d3ee", soft: "#0e7490" },
        violet: { DEFAULT: "#a78bfa" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
