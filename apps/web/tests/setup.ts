import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { createElement } from "react";
import type { ComponentProps } from "react";
import { afterEach, vi } from "vitest";

afterEach(() => cleanup());

vi.mock("next/image", () => ({
  default: (props: ComponentProps<"img">) => createElement("img", props),
}));
