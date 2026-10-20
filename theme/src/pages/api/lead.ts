import type { APIRoute } from "astro";

// This route is server-rendered (Vercel Serverless Function)
export const prerender = false;

interface GHLContactPayload {
  locationId: string;
  firstName: string;
  lastName: string;
  email: string;
  tags: string[];
}

export const POST: APIRoute = async ({ request }) => {
  const GHL_API_KEY = import.meta.env.GHL_API_KEY;
  const GHL_LOCATION_ID = import.meta.env.GHL_LOCATION_ID;

  if (!GHL_API_KEY || !GHL_LOCATION_ID) {
    console.error("Missing GHL_API_KEY or GHL_LOCATION_ID env vars");
    return new Response(
      JSON.stringify({ success: false, error: "Server misconfiguration" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }

  // Parse incoming form data
  let name = "";
  let email = "";

  const contentType = request.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    const body = await request.json();
    name = body.name ?? "";
    email = body.email ?? "";
  } else {
    const formData = await request.formData();
    name = (formData.get("name") as string) ?? "";
    email = (formData.get("email") as string) ?? "";
  }

  // Basic validation
  if (!name || !email) {
    return new Response(
      JSON.stringify({ success: false, error: "Name and email are required" }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  const nameParts = name.trim().split(/\s+/);
  const firstName = nameParts[0] ?? "";
  const lastName = nameParts.slice(1).join(" ");

  const payload: GHLContactPayload = {
    locationId: GHL_LOCATION_ID,
    firstName,
    lastName,
    email: email.trim(),
    tags: ["website-lead-generation"],
  };

  try {
    const ghlRes = await fetch(
      "https://services.leadconnectorhq.com/contacts/upsert",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GHL_API_KEY}`,
          "Content-Type": "application/json",
          Version: "2021-07-28",
        },
        body: JSON.stringify(payload),
      },
    );

    if (!ghlRes.ok) {
      const errText = await ghlRes.text();
      let detail = "Failed to submit. Please try again.";
      try {
        const errJson = JSON.parse(errText);
        detail = errJson.message || errJson.error || errText;
      } catch (e) {
        detail = errText;
      }

      console.error(`GHL API error ${ghlRes.status}:`, errText);
      return new Response(
        JSON.stringify({
          success: false,
          error: detail,
        }),
        {
          status: ghlRes.status,
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    const result = await ghlRes.json();
    console.log("GHL lead created:", result?.contact?.id);

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Lead generation error:", err);
    return new Response(
      JSON.stringify({
        success: false,
        error: "An unexpected error occurred.",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
};
