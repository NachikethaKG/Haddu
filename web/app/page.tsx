"use client";

import dynamic from "next/dynamic";
import { ShieldAlert, Radio, Compass, MapPin } from "lucide-react";

// Dynamic import with SSR disabled to prevent WebGL/window errors on server
const Map = dynamic(() => import("@/components/Map"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-950 text-slate-400 gap-3">
      <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <p className="text-sm font-medium tracking-wide">Initializing Satellite Map Canvas...</p>
    </div>
  ),
});

export default function HomePage() {
  return (
    <main className="relative w-full h-full overflow-hidden">
      {/* Full-viewport MapLibre GL Canvas */}
      <Map />

      {/* Brand Header & Responsive Overlay Shell */}
      <header className="absolute top-4 left-4 z-10 flex items-center gap-3">
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-2xl px-4 py-3 shadow-2xl flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center shadow-lg shadow-emerald-950/50">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold tracking-wider text-white uppercase">
                HADDU <span className="text-emerald-400 font-mono text-xs font-normal">v1.0</span>
              </h1>
              <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                ACTIVE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium flex items-center gap-1.5 mt-0.5">
              <span>Thirthahalli Forest Encroachment Monitor</span>
              <span className="text-slate-600">•</span>
              <span className="flex items-center gap-0.5 text-slate-400">
                <MapPin className="w-3 h-3 text-emerald-400/80" />
                Western Ghats, KA
              </span>
            </p>
          </div>
        </div>
      </header>

      {/* Coordinates / Status Pill (Top Center) */}
      <div className="hidden md:flex absolute top-4 left-1/2 -translate-x-1/2 z-10">
        <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800/80 rounded-full px-4 py-1.5 shadow-xl flex items-center gap-3 text-xs text-slate-300 font-mono">
          <span className="flex items-center gap-1.5 text-slate-400">
            <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
            <span>AOI: Thirthahalli Taluk</span>
          </span>
          <span className="text-slate-700">|</span>
          <span className="flex items-center gap-1 text-slate-300">
            <Compass className="w-3.5 h-3.5 text-emerald-400" />
            <span>13.7000° N, 75.2500° E</span>
          </span>
        </div>
      </div>
    </main>
  );
}
