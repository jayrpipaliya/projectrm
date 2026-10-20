<h1 align=center>Automark - Free Astro + GoHighLevel SaaS Website Template</h1>
<p align=center> Automark is a free Astro SaaS theme specifically crafted for GoHighLevel White-Label SaaS resellers and agencies.
</p>

<h2 align="center"> 
  <a target="_blank" href="https://automark-astro.vercel.app/" rel="nofollow">👀Demo</a> | 
  <a target="_blank" href="https://pagespeed.web.dev/report?url=https%3A%2F%2Fautomark-astro.vercel.app%2F&form_factor=desktop">Page Speed (100%)🚀</a>
</h2>

<p align=center>
  <a href="https://github.com/withastro/astro/releases/tag/astro%406.1.5" alt="Contributors">
    <img src="https://img.shields.io/static/v1?label=ASTRO&message=6.1.5&color=000&logo=astro" />
  </a>

  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/themefisher/automark-astro" alt="license"></a>

  <img src="https://img.shields.io/github/languages/code-size/themefisher/automark-astro" alt="code size">

  <a href="https://github.com/themefisher/automark-astro/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/themefisher/automark-astro" alt="contributors"></a>
</p>

<img width="1600" height="900" alt="automark-astro-gohighlevel-template" src="https://github.com/user-attachments/assets/dc93ae19-4e8a-46e3-b5b6-41c7bd46870c" />

While most resellers use the same stock templates, Automark gives you a modern, bespoke look to differentiate your brand. It comes pre-integrated with **GoHighLevel** for CRM/Lead management and **Stripe** for subscription handling, allowing you to launch a professional SaaS presence in minutes.

The template is **Agent-Ready**, featuring a dedicated .agents/skills directory designed to work seamlessly with agentic IDEs like Antigravity, Cursor, VS Code, and Claude Code. It follows industry-leading patterns for Astro, GoHighLevel, and Stripe best practices, making it as easy for an AI agent to maintain as it is for a human.

## 🔑 Key Features

* **Native GoHighLevel integration** for leads, contacts, conversations, calendar booking, and more.
* **Stripe subscription suite** with automated checkout sessions and GHL contact tagging via webhooks
* **10+ conversion-focused pages** covering every section a SaaS reseller website needs
* **Agentic IDE support** with a pre-configured `.agents/skills` directory for Cursor, Claude Code, and VS Code
* **100 Google** [**PageSpeed score**](https://pagespeed.web.dev/analysis/https-automark-astro-vercel-app/xtvggyeeh3?form_factor=desktop) (Desktop)
* **Built with Astro 6 and Tailwind CSS 4** for minimal JavaScript output and fast load times
* **MDX support** for blog posts, case studies, and careers pages
* **Automatic OG images and sitemap generation** with schema-friendly markup throughout
* **Interactive pricing table** with monthly/yearly toggle built in

## 📄 Pre-Designed Pages

1. **Home**: High-conversion landing page with automation focus.
2. **About**: Showcase your team, mission, and company values.
3. **Features**: Detailed breakdown of automation and growth tools.
4. **Integrations**: Display your ecosystem with a sleek grid layout.
5. **Pricing**: Interactive pricing tiers with monthly/yearly switching.
6. **Blog**: Full-featured blog with category and single post support.
7. **Careers**: Job listings with individual position pages.
8. **Case Studies**: Portfolio of client success stories.
9. **Contact**: GHL-integrated form and calendar booking.
10. **Legal Pages**: Professional Privacy Policy and Terms templates.
11. **404 Page**: Custom error page.

## ⚙️Installation

### 🔧Install prerequisites

- **Node.js:** [Install Node.js LTS](https://nodejs.org/en/download/)
- **Package Manager:** `npm`, `yarn`, or `pnpm` (Yarn recommended).

### 🖥️Local setup

1.  👉 **Install dependencies**
    ```bash
    yarn install
    ```

2.  👉 **Configure Environment Variables**
    Copy `.env.example` to `.env` and fill in your integration keys.
    ```bash
    cp .env.example .env
    ```

3.  👉 **Run locally**
    ```bash
    yarn dev
    ```

## 🔌 Integration Setup guide

Automark works out-of-the-box with GoHighLevel and Stripe. Follow these steps to activate the automation features:

### 1. GoHighLevel (GHL) Setup
This project uses GHL to capture leads from contact forms, track checkout initiations, and mark customers as paid upon successful Stripe transactions.

- **Location ID**: Navigate to **Settings** -> **Business Info** in your GHL sub-account and copy the **Location ID**.
- **Private API Key**: Go to **Settings** -> **Integrations** -> **Private Integration**. Generate/Copy your API Key.
  - Create a **"price"** field (Text/Monetary) to track Stripe purchases.
  - *Note: Ensure the Unique Key in GHL matches `"price"` exactly as used in the API routes.*
- **Calendar Meeting**: Find your calendar in **Calendars**, copy the **Scheduling Link**, and paste it into `src/content/contact/-index.md`.

### 2. Stripe Setup
Stripe handles subscriptions and one-time payments, syncing status back to GHL via webhooks.

- **API Keys**: Get your **Secret key** (`sk_...`) from **Developers** -> **API keys**.
- **Configure Webhooks**: 
  - Add an endpoint in **Developers** -> **Webhooks** pointing to `https://your-domain.com/api/webhooks/stripe`.
  - Select the `checkout.session.completed` event.
  - Copy the **Signing secret** (`whsec_...`) to your `.env` file.

### 3. Workflow Overview
- **`/api/lead`**: Captures leads and assigns the `website-lead-generation` tag.
- **`/api/contact`**: Captures form data, assigns `website-contact-form`, and sends the message to the GHL **Conversations** tab.
- **`/api/checkout`**: Initiates a Stripe subscription and tags the contact as `checkout-initiated`.
- **Stripe Webhook**: Updates the GHL contact to `paid-customer` and logs the price upon successful payment.

## 🔨Production Build

```bash
yarn build
```

## 📝 Edit Content with CMS

This template comes pre-configured with [**Sitepins**](https://sitepins.com/?aff=tfgithub), a Git-based Headless CMS designed for seamless content management.

**How to get started:**

Click the Edit with Sitepins button below to start editing your content visually.

<a target="_blank" href="https://app.sitepins.com/new/clone?name=Automark%20Astro&repository=https://github.com/themefisher/automark-astro/?aff=tfgithub">
  <img src="https://sitepins.com/button.svg" alt="Edit with Sitepins">
</a>

## 🐞Reporting Issues

We use GitHub Issues as the official bug tracker. Please Search [existing issues](https://github.com/themefisher/automark-astro/issues) before opening a new one.

## 📄License

Copyright (c) 2016 - Present, Designed & Developed by [Themefisher](https://themefisher.com)

- **Code License:** Released under the [MIT](LICENSE) license.
- **Image license:** Demonstration purposes only.
