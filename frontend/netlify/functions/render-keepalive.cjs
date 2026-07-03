const DEFAULT_HEALTH_URL = "https://f1-tire-deg-predictor.onrender.com/health";

exports.handler = async () => {
  const healthUrl = process.env.RENDER_HEALTH_URL || DEFAULT_HEALTH_URL;
  const startedAt = Date.now();

  const response = await fetch(healthUrl, {
    headers: {
      "user-agent": "f1-tire-deg-netlify-keepalive/1.0",
    },
  });

  const body = await response.text();
  const elapsedMs = Date.now() - startedAt;

  if (!response.ok) {
    throw new Error(`Render keepalive failed with ${response.status}: ${body}`);
  }

  console.log(`Render keepalive succeeded in ${elapsedMs}ms: ${body}`);

  return {
    statusCode: 200,
    body: JSON.stringify({
      ok: true,
      elapsed_ms: elapsedMs,
      status: response.status,
    }),
  };
};
