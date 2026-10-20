import type { CollectionEntry } from "astro:content";

// ─── Content Collection Data Types ───────────────────────────────────────────
// Derived directly from the Zod schemas in content.config.ts — always in sync.

export type Blog = CollectionEntry<"blog">["data"];
export type BlogIndex = CollectionEntry<"blogIndex">["data"];
export type About = CollectionEntry<"about">["data"];
export type Contact = CollectionEntry<"contact">["data"];
export type Homepage = CollectionEntry<"homepage">["data"];
export type Features = CollectionEntry<"features">["data"];
export type Pricing = CollectionEntry<"pricing">["data"];
export type CaseStudy = CollectionEntry<"caseStudy">["data"];
export type Careers = CollectionEntry<"careers">["data"];
export type Integrations = CollectionEntry<"integrations">["data"];

// ─── Section Data Types ───────────────────────────────────────────────────────

export type CtaSection = CollectionEntry<"ctaSection">["data"];
export type FaqSection = CollectionEntry<"faqSection">["data"];
export type BrandsSection = CollectionEntry<"brandsSection">["data"];
export type OurStorySection = CollectionEntry<"ourStorySection">["data"];
export type TestimonialSection = CollectionEntry<"testimonialSection">["data"];
export type ComparisonRowSection =
  CollectionEntry<"comparisonRowSection">["data"];
export type BusinessNeedsSection =
  CollectionEntry<"businessNeedsSection">["data"];

// ─── Config File Types ────────────────────────────────────────────────────────

export interface NavLink {
  name: string;
  url: string;
}

export interface Menu {
  main: NavLink[];
  footer_col_1_title: string;
  footer_col_2_title: string;
  footer_col_3_title: string;
  footer_primary: NavLink[];
  footer_resource: NavLink[];
  footer_legal: NavLink[];
}

export interface SocialLink {
  name: string;
  icon: string;
  link: string;
}

export interface SocialType {
  main: SocialLink[];
}

// ─── Shared Component Prop Types ─────────────────────────────────────────────

export interface FeatureItem {
  logo: string;
  title: string;
  is_starred: boolean;
}

export interface OurFeaturesSection {
  badge?: string;
  title: string;
  /** Used by homepage collection */
  content?: string;
  /** Used by about collection (core_values) */
  subtitle?: string;
  items: FeatureItem[];
}
