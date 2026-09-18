const fallbackMinimum = 100;
const fallbackMaximum = 20_000;

function publicInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

export const minimumArticleLength = publicInteger(
  process.env.NEXT_PUBLIC_MIN_ARTICLE_LENGTH,
  fallbackMinimum,
);

export const maximumArticleLength = Math.max(
  minimumArticleLength,
  publicInteger(process.env.NEXT_PUBLIC_MAX_ARTICLE_LENGTH, fallbackMaximum),
);
