import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "VeriScope | Evidence-aware news analysis",
  description: "Analyse news text, inspect model signals, and review current-source evidence.",
  icons: {
    icon: "/VeriScope.png",
    apple: "/VeriScope.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
