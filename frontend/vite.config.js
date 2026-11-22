import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// VITE_API_URL (see .env.example) is the gateway's base URL, read at build time.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: { environment: "node" },
});
