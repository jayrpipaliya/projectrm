// sort by date
export const sortByDate = <T extends { data: { date?: Date } }>(
  array: T[],
): T[] => {
  return array.sort(
    (a, b) =>
      new Date(b.data.date ?? 0).valueOf() -
      new Date(a.data.date ?? 0).valueOf(),
  );
};

// sort by weight
export const sortByWeight = <T extends { data: { weight?: number } }>(
  array: T[],
): T[] => {
  const withWeight = array.filter((item) => item.data.weight !== undefined);
  const withoutWeight = array.filter((item) => item.data.weight === undefined);
  const sortedWeightedArray = withWeight.sort(
    (a, b) => (a.data.weight ?? 0) - (b.data.weight ?? 0),
  );
  return [...new Set([...sortedWeightedArray, ...withoutWeight])];
};
