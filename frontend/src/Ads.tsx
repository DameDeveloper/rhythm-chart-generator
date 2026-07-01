import { useEffect, useRef } from 'react'

// Google AdSense integration.
//
// Ads are only loaded when VITE_ADSENSE_CLIENT is set at build time, so the
// desktop/exe build and local dev show no ads (and never hit AdSense's script).
// Set these in frontend/.env (see .env.example):
//
//   VITE_ADSENSE_CLIENT=ca-pub-XXXXXXXXXXXXXXXX
//   VITE_ADSENSE_SLOT=XXXXXXXXXX
//
// The publisher (ca-pub-...) and slot ids come from your AdSense account.

const CLIENT = (import.meta.env.VITE_ADSENSE_CLIENT as string | undefined) || ''
const DEFAULT_SLOT = (import.meta.env.VITE_ADSENSE_SLOT as string | undefined) || ''

export const adsEnabled = CLIENT.startsWith('ca-pub-')

declare global {
  interface Window {
    adsbygoogle?: unknown[]
  }
}

let scriptLoaded = false

/** Inject the AdSense loader script once per page load. */
function ensureAdScript() {
  if (scriptLoaded || !adsEnabled || typeof document === 'undefined') return
  scriptLoaded = true
  const s = document.createElement('script')
  s.async = true
  s.crossOrigin = 'anonymous'
  s.src = `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${CLIENT}`
  document.head.appendChild(s)
}

interface AdSlotProps {
  slot?: string
  /** e.g. 'horizontal', 'rectangle'; passed to data-ad-format. */
  format?: string
  className?: string
  style?: React.CSSProperties
}

/**
 * A single responsive ad unit. Renders nothing unless AdSense is configured,
 * so it is safe to leave in the tree for every build.
 */
export function AdSlot({ slot, format = 'auto', className, style }: AdSlotProps) {
  const ref = useRef<HTMLModElement>(null)
  const pushed = useRef(false)
  const adSlot = slot || DEFAULT_SLOT

  useEffect(() => {
    if (!adsEnabled || !adSlot || pushed.current) return
    ensureAdScript()
    try {
      ;(window.adsbygoogle = window.adsbygoogle || []).push({})
      pushed.current = true
    } catch {
      /* AdSense not ready yet; it will retry on next render */
    }
  }, [adSlot])

  if (!adsEnabled || !adSlot) return null

  return (
    <ins
      ref={ref}
      className={`adsbygoogle ${className || ''}`}
      style={{ display: 'block', ...style }}
      data-ad-client={CLIENT}
      data-ad-slot={adSlot}
      data-ad-format={format}
      data-full-width-responsive="true"
    />
  )
}
