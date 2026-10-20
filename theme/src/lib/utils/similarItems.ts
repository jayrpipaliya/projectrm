import type { CollectionEntry } from "astro:content";

type BlogEntry = CollectionEntry<"blog">;

// Returns blog posts related to the current post by categories and tags
const similarItems = (
  currentItem: BlogEntry,
  allItems: BlogEntry[],
): BlogEntry[] => {
  let categories: string[] = [];
  let tags: string[] = [];

  // set categories
  if (currentItem.data.categories.length > 0) {
    categories = currentItem.data.categories;
  }

  // set tags
  if (currentItem.data.tags.length > 0) {
    tags = currentItem.data.tags;
  }

  // filter by categories
  const filterByCategories = allItems.filter((item) =>
    categories.find((category) => item.data.categories.includes(category)),
  );

  // filter by tags
  const filterByTags = allItems.filter((item) =>
    tags.find((tag) => item.data.tags.includes(tag)),
  );

  // merged after filter
  const mergedItems = [...new Set([...filterByCategories, ...filterByTags])];

  // filter by slug
  const filterBySlug = mergedItems.filter(
    (product) => product.id !== currentItem.id,
  );

  return filterBySlug;
};

export default similarItems;
