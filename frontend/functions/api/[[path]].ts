// Cloudflare Pages Function: proxy /api/* to the backend container.
//
// The backend (main:app) exposes its routes at the root (/upload, /pipeline,
// /audio/..., /export, /health). The frontend calls them under /api/*, so we
// strip the leading /api and forward everything to BACKEND_URL.
//
// Set BACKEND_URL in the Cloudflare Pages project settings, e.g.
//   BACKEND_URL = https://rhythm-backend.example.com
//
// Keeping this same-origin means no CORS and cookies/analytics behave normally.

interface Env {
  BACKEND_URL: string
}

export const onRequest: PagesFunction<Env> = async (context) => {
  const { request, env } = context
  const base = (env.BACKEND_URL || '').replace(/\/+$/, '')
  if (!base) {
    return new Response('BACKEND_URL is not configured', { status: 502 })
  }

  const incoming = new URL(request.url)
  // '/api/upload?x=1' -> '/upload?x=1'
  const forwardPath = incoming.pathname.replace(/^\/api/, '') || '/'
  const target = base + forwardPath + incoming.search

  // Rebuild the request so streamed uploads/downloads pass through untouched.
  const init: RequestInit = {
    method: request.method,
    headers: request.headers,
    body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
    redirect: 'manual',
  }

  const resp = await fetch(target, init)
  // Clone headers so we can drop hop-by-hop entries if present.
  const headers = new Headers(resp.headers)
  headers.delete('transfer-encoding')
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers,
  })
}
