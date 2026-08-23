/**
 * Build an API URL from the deployment-provided base URL.
 *
 * The client deliberately has no localhost fallback: a missing Vercel
 * variable should fail clearly instead of sending production traffic to a
 * developer machine.
 */
export function apiUrl(path: string): string {
  const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

  if (!configuredApiUrl) {
    throw new Error("NEXT_PUBLIC_API_URL is not configured");
  }

  let baseUrl: URL;
  try {
    baseUrl = new URL(configuredApiUrl);
  } catch {
    throw new Error("NEXT_PUBLIC_API_URL must be a valid absolute URL");
  }

  if (baseUrl.protocol !== "http:" && baseUrl.protocol !== "https:") {
    throw new Error("NEXT_PUBLIC_API_URL must use http or https");
  }

  if (baseUrl.search || baseUrl.hash) {
    throw new Error("NEXT_PUBLIC_API_URL must not include a query or hash");
  }

  // Keep an optional API prefix such as /api/v1 while preventing URL's
  // resolution rules from treating the first route segment as a replacement.
  baseUrl.pathname = `${baseUrl.pathname.replace(/\/+$/, "")}/`;

  const normalizedPath = path.trim().replace(/^\/+/, "");
  if (
    /^[a-z][a-z\d+.-]*:/i.test(normalizedPath) ||
    normalizedPath.startsWith("//")
  ) {
    throw new Error("API paths must be relative URLs");
  }

  return new URL(normalizedPath, baseUrl).toString();
}
