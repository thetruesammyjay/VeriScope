import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SiteHeader } from "@/components/site-header";

describe("SiteHeader", () => {
  it("opens and closes the mobile navigation with the hamburger button", () => {
    render(<SiteHeader />);

    const toggle = screen.getByRole("button", { name: "Open navigation menu" });
    const mobileNavigation = screen.getByRole("navigation", {
      name: "Mobile navigation",
    }).parentElement;

    expect(mobileNavigation).not.toHaveClass("is-open");

    fireEvent.click(toggle);

    expect(mobileNavigation).toHaveClass("is-open");
    expect(
      screen.getByRole("button", { name: "Close navigation menu" }),
    ).toHaveAttribute("aria-expanded", "true");

    fireEvent.click(
      mobileNavigation!.querySelector('a[href="#responsible-use"]')!,
    );

    expect(mobileNavigation).not.toHaveClass("is-open");
  });
});
