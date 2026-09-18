"use client";

import Image from "next/image";
import { useState } from "react";

const navigation = [
  { href: "#analyse", label: "Analyse" },
  { href: "#how-it-works", label: "How it works" },
  { href: "#responsible-use", label: "Responsible use" },
];

export function SiteHeader() {
  const [isOpen, setIsOpen] = useState(false);

  const closeMenu = () => setIsOpen(false);

  return (
    <header className="site-header">
      <div className="header-inner">
        <a className="brand" href="#top" aria-label="VeriScope home" onClick={closeMenu}>
          <Image src="/VeriScope.png" alt="VeriScope" width={244} height={57} priority />
        </a>

        <nav className="desktop-nav" aria-label="Primary navigation">
          {navigation.map((item) => (
            <a key={item.href} href={item.href}>{item.label}</a>
          ))}
        </nav>

        <a className="header-cta" href="#analyse">Analyse an article <span aria-hidden="true">↗</span></a>

        <button
          className="menu-button"
          type="button"
          aria-label={isOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={isOpen}
          aria-controls="mobile-navigation"
          onClick={() => setIsOpen((open) => !open)}
        >
          <span /> <span />
        </button>
      </div>

      <div id="mobile-navigation" className={`mobile-nav ${isOpen ? "is-open" : ""}`}>
        <nav aria-label="Mobile navigation">
          {navigation.map((item) => (
            <a key={item.href} href={item.href} onClick={closeMenu}>{item.label}</a>
          ))}
          <a className="mobile-nav-cta" href="#analyse" onClick={closeMenu}>Analyse an article <span aria-hidden="true">↗</span></a>
        </nav>
      </div>
    </header>
  );
}
