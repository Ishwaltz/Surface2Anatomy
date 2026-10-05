/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F5F5F0",
        surface: "#FFFFFF",
        border: "#D8DCCF",
        primary: {
          DEFAULT: "#66734B",
          dark: "#465133",
          light: "#8B9A6D",
        },
        text: {
          main: "#1F241B",
          muted: "#636E58",
        },
        accent: {
          green: "#4E774A",
          amber: "#C88A2D",
          red: "#B34A3E",
          blue: "#3C6E71",
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
