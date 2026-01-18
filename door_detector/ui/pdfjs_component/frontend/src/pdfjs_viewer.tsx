import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ComponentProps, Streamlit } from "streamlit-component-lib";

import { getDocument, GlobalWorkerOptions, PDFDocumentProxy, PDFPageProxy } from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = workerSrc;

type BBox = [number, number, number, number];

type OverlayDoor = {
  id: string;
  // In PDF coordinate space (bottom-left origin, points).
  bbox_pdf_xyxy: BBox;
};

type Candidate = {
  id: string;
  bbox_pdf_xyxy: BBox;
};

type DoorState = {
  confirmed_ids?: string[];
  deleted_ids?: string[];
};

type ManualOverlayPayload = {
  manual_additions?: Array<{
    drawn_bbox_pdf_xyxy: BBox;
    snapped_bbox_pdf_xyxy?: BBox | null;
    snapped_candidate_id?: string | null;
    iou?: number | null;
  }>;
  unmatched_manual_boxes?: Array<{
    bbox_pdf_xyxy: BBox;
    note?: string;
  }>;
};

type ViewerEvent =
  | { type: "door_click"; event_id: string; door_id: string; ts: number }
  | {
      type: "draw_rect";
      event_id: string;
      bbox_pdf_xyxy: BBox;
      snapped_candidate_id: string | null;
      iou: number | null;
      snapped_bbox_pdf_xyxy: BBox | null;
      ts: number;
    };

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

function easeInOut(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function bboxCenter(b: BBox) {
  return { x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2 };
}

function normalizeBBox(b: BBox): BBox {
  const x0 = Math.min(b[0], b[2]);
  const y0 = Math.min(b[1], b[3]);
  const x1 = Math.max(b[0], b[2]);
  const y1 = Math.max(b[1], b[3]);
  return [x0, y0, x1, y1];
}

function pdfBBoxToViewportBBox(vp: any, pdfBBox: BBox): BBox {
  // Use point conversion rather than convertToViewportRectangle to avoid
  // confusion about output ordering across pdf.js versions.
  const [x0, y0, x1, y1] = normalizeBBox(pdfBBox);
  const p0 = vp.convertToViewportPoint(x0, y0);
  const p1 = vp.convertToViewportPoint(x1, y1);
  return normalizeBBox([p0[0], p0[1], p1[0], p1[1]]);
}

function computeIoU(a: BBox, b: BBox): number {
  const [ax0, ay0, ax1, ay1] = normalizeBBox(a);
  const [bx0, by0, bx1, by1] = normalizeBBox(b);
  const ix0 = Math.max(ax0, bx0);
  const iy0 = Math.max(ay0, by0);
  const ix1 = Math.min(ax1, bx1);
  const iy1 = Math.min(ay1, by1);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  const aa = Math.max(0, ax1 - ax0) * Math.max(0, ay1 - ay0);
  const ba = Math.max(0, bx1 - bx0) * Math.max(0, by1 - by0);
  const denom = aa + ba - inter;
  if (!(denom > 0)) return 0;
  return inter / denom;
}

function computeIntersectionArea(a: BBox, b: BBox): number {
  const [ax0, ay0, ax1, ay1] = normalizeBBox(a);
  const [bx0, by0, bx1, by1] = normalizeBBox(b);
  const ix0 = Math.max(ax0, bx0);
  const iy0 = Math.max(ay0, by0);
  const ix1 = Math.min(ax1, bx1);
  const iy1 = Math.min(ay1, by1);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  return iw * ih;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function isLocalStorageAvailable(): boolean {
  try {
    const k = "__door_detector_ls_probe__";
    window.localStorage.setItem(k, "1");
    window.localStorage.removeItem(k);
    return true;
  } catch {
    return false;
  }
}

function randEventId(): string {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function PdfJsViewer(props: ComponentProps) {
  const args = props.args as any;

  const fileId = String(args.fileId ?? "");
  const height = Number(args.height ?? 650);
  const pdfHash = String(args.pdfHash ?? "");
  const pdfDataB64 = (args.pdfDataB64 ?? null) as string | null;
  const pageNumber = Number(args.pageNumber ?? 1);
  const unmatchedDebugRaw = String(args.unmatchedDebugRaw ?? "");

  const selectedDoorId = String(args.selectedDoorId ?? "");
  const focusSeq = Number(args.focusSeq ?? 0);
  const editMode = Boolean(args.editMode ?? false);
  const viewerDisplayMode = String(args.viewerDisplayMode ?? "all");

  const doorState = (args.doorState ?? {}) as DoorState;
  const manualOverlays = (args.manualOverlays ?? {}) as ManualOverlayPayload;

  const overlayDoors = (args.overlayDoors ?? []) as OverlayDoor[];
  const candidatePool = (args.candidatePool ?? []) as Candidate[];

  const rootRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  // Double-buffered canvases to avoid blank frames during re-render.
  // We render into the inactive canvas, then swap opacity.
  const canvasARef = useRef<HTMLCanvasElement | null>(null);
  const canvasBRef = useRef<HTMLCanvasElement | null>(null);
  const [activeCanvas, setActiveCanvas] = useState<"a" | "b">("a");
  const svgRef = useRef<SVGSVGElement | null>(null);

  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const pageRef = useRef<PDFPageProxy | null>(null);
  const viewportRef = useRef<any>(null); // PDF.js PageViewport at scale=1
  const lastLoadedHashRef = useRef<string | null>(null);

  const [pageSize, setPageSize] = useState<{ w: number; h: number } | null>(null);

  // Canvas render quality: we render once at baseline, and (optionally) re-render
  // at higher internal resolution when the user zooms in (zoom-driven only).
  const renderQualityRef = useRef<number>(1);
  const renderTimerRef = useRef<number | null>(null);
  const renderInFlightRef = useRef<boolean>(false);
  const activeCanvasRef = useRef<"a" | "b">("a");

  // Pan/zoom state (applied as a CSS transform on the content div).
  const stateKey = useMemo(() => `door_detector_pdfjs_state_${fileId}`, [fileId]);
  const scaleRef = useRef(1);
  const txRef = useRef(0);
  const tyRef = useRef(0);
  const baseScaleRef = useRef(1);
  const baseTxRef = useRef(0);
  const baseTyRef = useRef(0);
  const focusSeqRef = useRef(0);
  const resetBtnRef = useRef<HTMLButtonElement | null>(null);
  const [resetVisible, setResetVisible] = useState(false);
  const resetVisibleRef = useRef(false);

  // Selection for immediate feedback (server catches up on rerun).
  const localSelectedIdRef = useRef<string | null>(null);
  const lastSelectedIdRef = useRef<string | null>(null);
  const lastViewerDisplayRef = useRef<string | null>(null);
  const lastEditModeRef = useRef<boolean | null>(null);

  const confirmedSet = useMemo(() => new Set((doorState.confirmed_ids ?? []).map(String)), [doorState]);
  const deletedSet = useMemo(() => new Set((doorState.deleted_ids ?? []).map(String)), [doorState]);

  const emitEvent = useCallback((evt: ViewerEvent) => {
    Streamlit.setComponentValue(evt);
  }, []);

  // Be explicit about readiness. (Some Streamlit frontends can log
  // "unregistered ComponentInstance" if messages arrive before registration.)
  useEffect(() => {
    try {
      Streamlit.setComponentReady();
    } catch {
      // ignore
    }
  }, []);

  // Mirror legacy iframe behavior: print unmatched debug reports to the browser console.
  useEffect(() => {
    if (!unmatchedDebugRaw) return;
    try {
      // Print as normal lines so it’s easy to copy/paste without expanding groups.
      // Matches the legacy viewer's log labels.
      // eslint-disable-next-line no-console
      console.log("[door_detector] unmatched_debug_report raw", unmatchedDebugRaw);
      let obj: any = null;
      try {
        obj = JSON.parse(unmatchedDebugRaw);
      } catch {
        obj = null;
      }
      if (obj) {
        // eslint-disable-next-line no-console
        console.log("[door_detector] unmatched_debug_report parsed", obj);
      } else {
        // eslint-disable-next-line no-console
        console.warn("[door_detector] unmatched_debug_report parse_failed");
      }
    } catch {
      // ignore
    }
  }, [unmatchedDebugRaw]);

  const updateResetVisibility = useCallback(() => {
    const eps = 0.5;
    const atBase =
      Math.abs(scaleRef.current - baseScaleRef.current) < 0.0005 &&
      Math.abs(txRef.current - baseTxRef.current) < eps &&
      Math.abs(tyRef.current - baseTyRef.current) < eps;
    const visible = !atBase;
    if (visible === resetVisibleRef.current) return;
    resetVisibleRef.current = visible;
    setResetVisible(visible);
  }, []);

  const applyTransform = useCallback(() => {
    const content = contentRef.current;
    if (!content) return;
    const s = scaleRef.current;
    const tx = txRef.current;
    const ty = tyRef.current;
    content.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    try {
      sessionStorage.setItem(
        stateKey,
        JSON.stringify({ tx, ty, scale: s, focusSeq: focusSeqRef.current })
      );
    } catch {
      // ignore
    }
    updateResetVisibility();
  }, [stateKey, updateResetVisibility]);

  const fitToContainer = useCallback(() => {
    const root = rootRef.current;
    const ps = pageSize;
    if (!root || !ps) return;
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    if (!(cw > 0) || !(ch > 0)) return;

    const pad = 6;
    const cwPad = cw - pad * 2;
    const chPad = ch - pad * 2;
    if (!(cwPad > 0) || !(chPad > 0)) return;

    const s = clamp(Math.min(cwPad / ps.w, chPad / ps.h), 0.05, 20);
    const tx = (cw - ps.w * s) / 2;
    const ty = (ch - ps.h * s) / 2;

    scaleRef.current = s;
    txRef.current = tx;
    tyRef.current = ty;
    baseScaleRef.current = s;
    baseTxRef.current = tx;
    baseTyRef.current = ty;
    applyTransform();
  }, [applyTransform, pageSize]);

  const resetView = useCallback(() => {
    // Reset to the same "default view" used on initial load: fit-to-container and centered.
    // Also clear persisted state so a rerun/reload doesn't snap back to a non-default view.
    try {
      sessionStorage.removeItem(stateKey);
    } catch {
      // ignore
    }
    fitToContainer();
  }, [fitToContainer, stateKey]);

  const loadState = useCallback(() => {
    try {
      const raw = sessionStorage.getItem(stateKey);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      if (!obj) return null;
      if (!Number.isFinite(obj.tx) || !Number.isFinite(obj.ty) || !Number.isFinite(obj.scale)) return null;
      return obj as { tx: number; ty: number; scale: number; focusSeq?: number };
    } catch {
      return null;
    }
  }, [stateKey]);

  const applyInitialView = useCallback(() => {
    if (!pageSize) return;
    const saved = loadState();
    fitToContainer();
    if (saved) {
      scaleRef.current = clamp(saved.scale, 0.05, 20);
      txRef.current = saved.tx;
      tyRef.current = saved.ty;
      applyTransform();
    }
  }, [applyTransform, fitToContainer, loadState, pageSize]);

  // Frame height: keep stable so Streamlit doesn't thrash layout.
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  // Load PDF only when hash changes and data is provided.
  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!pdfHash) return;
      if (lastLoadedHashRef.current === pdfHash && pdfRef.current && pageRef.current) return;
      // Cache PDF bytes in localStorage keyed by hash to avoid re-sending large
      // base64 payloads on every Streamlit rerun (when possible).
      let data: Uint8Array | null = null;
      if (isLocalStorageAvailable()) {
        try {
          const lk = `door_detector_pdf_b64_${pdfHash}`;
          const cached = window.localStorage.getItem(lk);
          if (cached) {
            data = base64ToBytes(cached);
          } else if (pdfDataB64) {
            window.localStorage.setItem(lk, pdfDataB64);
            data = base64ToBytes(pdfDataB64);
          }
        } catch {
          data = null;
        }
      }
      if (!data) {
        if (!pdfDataB64) return;
        data = base64ToBytes(pdfDataB64);
      }

      const task = getDocument({ data });
      const pdf = await task.promise;
      const page = await pdf.getPage(pageNumber);
      if (cancelled) return;

      pdfRef.current = pdf;
      pageRef.current = page;
      lastLoadedHashRef.current = pdfHash;

      // Determine base page size in CSS pixels (scale=1).
      const vp = page.getViewport({ scale: 1 });
      viewportRef.current = vp;
      setPageSize({ w: vp.width, h: vp.height });
    }

    run().catch((err) => {
      // eslint-disable-next-line no-console
      console.error("[door_detector] pdfjs load failed", err);
    });

    return () => {
      cancelled = true;
    };
  }, [pdfDataB64, pdfHash, pageNumber]);

  // Render canvas when page changes (or first load). Zoom quality upgrades happen later.
  useEffect(() => {
    let cancelled = false;

    async function render() {
      const page = pageRef.current;
      const canvas = canvasARef.current;
      if (!page || !canvas || !pageSize) return;

      const dpr = window.devicePixelRatio || 1;
      const renderScale = 1 * dpr; // baseline; upgraded later
      const viewport = page.getViewport({ scale: renderScale });

      const ctx = canvas.getContext("2d", { alpha: false });
      if (!ctx) return;

      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      canvas.style.width = `${pageSize.w}px`;
      canvas.style.height = `${pageSize.h}px`;

      const task = page.render({ canvasContext: ctx, viewport });
      await task.promise;
      if (cancelled) return;

      renderQualityRef.current = 1;
      activeCanvasRef.current = "a";
      setActiveCanvas("a");
      // After first render, apply fit + saved state.
      applyInitialView();
    }

    render().catch((err) => {
      // eslint-disable-next-line no-console
      console.error("[door_detector] pdfjs render failed", err);
    });

    return () => {
      cancelled = true;
    };
  }, [applyInitialView, pageSize]);

  const renderCanvasAtQuality = useCallback(async (quality: number) => {
    const page = pageRef.current;
    if (!page || !pageSize) return;
    if (!(quality >= 1)) quality = 1;
    if (renderInFlightRef.current) return;
    if (quality === renderQualityRef.current) return;

    const dpr = window.devicePixelRatio || 1;
    // Cap quality so we don't exceed typical GPU/Canvas limits.
    // (Prevent huge allocations at extreme zoom + large pages.)
    const MAX_DIM = 16384;
    const maxByDim = Math.max(
      1,
      Math.floor(Math.min(MAX_DIM / (pageSize.w * dpr), MAX_DIM / (pageSize.h * dpr)))
    );
    quality = Math.min(quality, maxByDim);
    if (quality === renderQualityRef.current) return;

    const nextCanvasKey: "a" | "b" = activeCanvasRef.current === "a" ? "b" : "a";
    const nextCanvas = nextCanvasKey === "a" ? canvasARef.current : canvasBRef.current;
    if (!nextCanvas) return;

    renderInFlightRef.current = true;
    try {
      const viewport = page.getViewport({ scale: quality * dpr });
      const ctx = nextCanvas.getContext("2d", { alpha: false });
      if (!ctx) return;

      // Resize the inactive canvas (the active one stays visible until we swap).
      nextCanvas.width = Math.floor(viewport.width);
      nextCanvas.height = Math.floor(viewport.height);
      nextCanvas.style.width = `${pageSize.w}px`;
      nextCanvas.style.height = `${pageSize.h}px`;

      const task = page.render({ canvasContext: ctx, viewport });
      await task.promise;
      // Give the browser a frame to commit the newly-rendered canvas before we
      // swap visibility. This avoids any transient "blank/grey" composite.
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      renderQualityRef.current = quality;

      // Swap active canvas without blending (no crossfade).
      activeCanvasRef.current = nextCanvasKey;
      setActiveCanvas(nextCanvasKey);
    } finally {
      renderInFlightRef.current = false;
    }
  }, [pageSize]);

  const scheduleQualityRender = useCallback(() => {
    if (!pageSize) return;
    const z = baseScaleRef.current > 0 ? scaleRef.current / baseScaleRef.current : scaleRef.current;
    // More resolution steps at higher zoom; actual cap is enforced in renderCanvasAtQuality.
    //
    // IMPORTANT: Use hysteresis so we don't bounce around thresholds (which can cause
    // repeated rerenders + visible flashes).
    const q0 = renderQualityRef.current;
    let desired = q0;

    // Upgrade thresholds.
    if (desired < 2 && z >= 2.2) desired = 2;
    if (desired < 3 && z >= 3.6) desired = 3;
    if (desired < 4 && z >= 5.5) desired = 4;
    if (desired < 5 && z >= 8.0) desired = 5;

    // Downgrade thresholds (lower than upgrade to create hysteresis).
    if (desired === 5 && z < 7.2) desired = 4;
    if (desired === 4 && z < 5.0) desired = 3;
    if (desired === 3 && z < 3.2) desired = 2;
    if (desired === 2 && z < 1.9) desired = 1;

    if (desired === q0) return;

    // Upgrades feel best quickly; downgrades can wait a bit longer to avoid churn.
    const delayMs = desired > q0 ? 220 : 520;
    if (renderTimerRef.current) window.clearTimeout(renderTimerRef.current);
    renderTimerRef.current = window.setTimeout(() => {
      renderTimerRef.current = null;
      void renderCanvasAtQuality(desired);
    }, delayMs);
  }, [pageSize, renderCanvasAtQuality]);

  // ResizeObserver: keep base fit-to-container updated when user hasn't deviated.
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const ro = new ResizeObserver(() => {
      const eps = 0.5;
      const s = scaleRef.current;
      const tx = txRef.current;
      const ty = tyRef.current;
      const atBase =
        Math.abs(s - baseScaleRef.current) < 0.0005 &&
        Math.abs(tx - baseTxRef.current) < eps &&
        Math.abs(ty - baseTyRef.current) < eps;
      if (atBase) fitToContainer();
    });
    ro.observe(root);
    return () => ro.disconnect();
  }, [fitToContainer]);

  const clearSvgLayer = useCallback((g: SVGGElement | null) => {
    if (!g) return;
    while (g.firstChild) g.removeChild(g.firstChild);
  }, []);

  const ensureLayer = useCallback((id: string): SVGGElement | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    let g = svg.querySelector<SVGGElement>(`#${id}`);
    if (g) return g;
    g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("id", id);
    g.setAttribute("shape-rendering", "crispEdges");
    svg.appendChild(g);
    return g;
  }, []);

  const drawBox = useCallback(
    (layer: SVGGElement | null, bbox: BBox, stroke: string, strokeWidth: number, dash: string | null, opacity: number) => {
      if (!layer) return;
      const q = (v: number) => Math.round(v * 2) / 2;
      const [x0, y0, x1, y1] = normalizeBBox(bbox);
      const w = q(Math.max(0, x1 - x0));
      const h = q(Math.max(0, y1 - y0));
      if (!(w > 0) || !(h > 0)) return;

      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", String(q(x0)));
      r.setAttribute("y", String(q(y0)));
      r.setAttribute("width", String(w));
      r.setAttribute("height", String(h));
      r.setAttribute("fill", "none");
      r.setAttribute("stroke", stroke);
      r.setAttribute("stroke-width", String(strokeWidth));
      r.setAttribute("stroke-linecap", "square");
      r.setAttribute("shape-rendering", "crispEdges");
      r.setAttribute("vector-effect", "non-scaling-stroke");
      if (dash) r.setAttribute("stroke-dasharray", dash);
      if (Number.isFinite(opacity)) r.setAttribute("stroke-opacity", String(opacity));
      (r.style as any).pointerEvents = "none";
      layer.appendChild(r);
    },
    []
  );

  const applyDoorStyles = useCallback(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const selectedId = localSelectedIdRef.current || (selectedDoorId ? selectedDoorId : null);
    const rects = svg.querySelectorAll<SVGRectElement>("rect[data-door-id]");
    for (const r of rects) {
      const did = r.getAttribute("data-door-id");
      if (!did) continue;

      const isSelected = !!selectedId && did === selectedId;
      const isDeleted = deletedSet.has(did);

      let visible = true;
      if (viewerDisplayMode === "off") visible = false;
      else if (viewerDisplayMode === "selected") visible = isSelected;
      if (isDeleted) visible = false;

      if (!visible) {
        (r.style as any).display = "none";
        (r.style as any).pointerEvents = "none";
        continue;
      }

      (r.style as any).display = "";
      (r.style as any).pointerEvents = "all";

      if (isSelected) {
        r.setAttribute("stroke", "#ff4b4b");
        r.setAttribute("stroke-width", "3");
        try {
          svg.appendChild(r);
        } catch {
          // ignore
        }
      } else if (confirmedSet.has(did)) {
        r.setAttribute("stroke", "#00ff00");
        r.setAttribute("stroke-width", "2");
      } else {
        r.setAttribute("stroke", "#ffa500");
        r.setAttribute("stroke-width", "2");
      }
    }
  }, [confirmedSet, deletedSet, selectedDoorId, viewerDisplayMode]);

  const renderOverlays = useCallback(() => {
    const svg = svgRef.current;
    if (!svg || !pageSize) return;
    const vp = viewportRef.current;
    if (!vp) return;

    // 1) Base door rectangles (interactive).
    // We redraw these each time props change for simplicity; they are small.
    // If this becomes hot, we can diff by id.
    // Keep them in their own group so edit overlays can layer above.
    const doorsLayer = ensureLayer("pz_doors");
    clearSvgLayer(doorsLayer);
    for (const d of overlayDoors) {
      if (!d || !d.id || !d.bbox_pdf_xyxy) continue;
      const [x0, y0, x1, y1] = pdfBBoxToViewportBBox(vp, d.bbox_pdf_xyxy);
      const w = Math.max(0, x1 - x0);
      const h = Math.max(0, y1 - y0);
      if (!(w > 0) || !(h > 0)) continue;
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", x0.toFixed(2));
      r.setAttribute("y", y0.toFixed(2));
      r.setAttribute("width", w.toFixed(2));
      r.setAttribute("height", h.toFixed(2));
      r.setAttribute("fill", "none");
      r.setAttribute("stroke", "#ffa500");
      r.setAttribute("stroke-width", "2");
      r.setAttribute("vector-effect", "non-scaling-stroke");
      r.setAttribute("data-door-id", String(d.id));
      r.setAttribute("data-x", x0.toFixed(2));
      r.setAttribute("data-y", y0.toFixed(2));
      r.setAttribute("data-w", w.toFixed(2));
      r.setAttribute("data-h", h.toFixed(2));
      r.setAttribute("style", "pointer-events: all; cursor: pointer;");
      doorsLayer?.appendChild(r);
    }

    // 2) Manual overlays (edit-mode only).
    const manualLayer = ensureLayer("pz_manual");
    const tempLayer = ensureLayer("pz_temp");
    if (!editMode) {
      clearSvgLayer(manualLayer);
      clearSvgLayer(tempLayer);
      return;
    }

    clearSvgLayer(manualLayer);
    // Once server overlays update, drop any client-only temp boxes to avoid duplicates.
    clearSvgLayer(tempLayer);

    const manual = manualOverlays.manual_additions ?? [];
    const unmatched = manualOverlays.unmatched_manual_boxes ?? [];
    for (const m of manual) {
      drawBox(manualLayer, pdfBBoxToViewportBBox(vp, m.drawn_bbox_pdf_xyxy), "rgb(0,255,255)", 2, "6,4", 0.47);
      if (m.snapped_bbox_pdf_xyxy) {
        drawBox(manualLayer, pdfBBoxToViewportBBox(vp, m.snapped_bbox_pdf_xyxy), "rgb(0,255,0)", 3, "4,3", 0.77);
      }
    }
    for (const u of unmatched) {
      drawBox(manualLayer, pdfBBoxToViewportBBox(vp, u.bbox_pdf_xyxy), "rgb(255,0,255)", 2, "6,4", 0.63);
    }
  }, [candidatePool, clearSvgLayer, drawBox, editMode, ensureLayer, manualOverlays, overlayDoors, pageSize]);

  const snapCandidateForDrawPdf = useCallback(
    (drawnPdf: BBox) => {
      // Mirror the legacy snap rules: only overlap candidates, then pick best by IoU
      // (>= MIN_SNAP_IOU), else fall back to max intersection area, else coverage.
      const MIN_SNAP_IOU = 0.02;
      const MIN_CAND_COVERAGE = 0.2;
      const norm = normalizeBBox(drawnPdf);
      const overlap: Array<{ id: string; bbox: BBox; iou: number; inter: number; coverage: number }> = [];

      let bestIou = -1;
      let bestByIou: { id: string; bbox: BBox; iou: number } | null = null;
      let bestInter = -1;
      let bestByInter: { id: string; bbox: BBox; iou: number; inter: number } | null = null;
      let bestCoverage = -1;
      let bestByCoverage: { id: string; bbox: BBox; iou: number; coverage: number; inter: number } | null = null;

      for (const r of candidatePool) {
        const did = r?.id ? String(r.id) : null;
        const cand = r?.bbox_pdf_xyxy && Array.isArray(r.bbox_pdf_xyxy) ? (r.bbox_pdf_xyxy as BBox) : null;
        if (!did || !cand || cand.length !== 4) continue;
        const iou = computeIoU(norm, cand);
        const inter = computeIntersectionArea(norm, cand);
        const candArea = Math.max(0, (cand[2] - cand[0])) * Math.max(0, (cand[3] - cand[1]));
        const coverage = candArea > 0 ? inter / candArea : 0;
        if (inter > 0) {
          overlap.push({ id: did, bbox: cand, iou, inter, coverage });
          if (iou > bestIou) {
            bestIou = iou;
            bestByIou = { id: did, bbox: cand, iou };
          }
          if (inter > bestInter) {
            bestInter = inter;
            bestByInter = { id: did, bbox: cand, iou, inter };
          }
          if (coverage > bestCoverage) {
            bestCoverage = coverage;
            bestByCoverage = { id: did, bbox: cand, iou, coverage, inter };
          }
        }
      }

      // Debug output: match the legacy viewer's "criteria hit/miss" style.
      // eslint-disable-next-line no-console
      console.log("[door_detector] snapCandidateForDrawPdf", {
        drawn: norm,
        candidates: candidatePool.length,
        overlapCandidates: overlap.length,
        thresholds: { MIN_SNAP_IOU, MIN_CAND_COVERAGE },
        bestByIoU: bestByIou ? { id: bestByIou.id, iou: bestByIou.iou } : null,
        bestByInter: bestByInter ? { id: bestByInter.id, inter: bestByInter.inter, iou: bestByInter.iou } : null,
        bestByCoverage: bestByCoverage
          ? { id: bestByCoverage.id, coverage: bestByCoverage.coverage, inter: bestByCoverage.inter, iou: bestByCoverage.iou }
          : null,
        overlapSample: overlap.slice(0, 3),
      });

      if (!overlap.length) {
        // eslint-disable-next-line no-console
        console.log("[door_detector] snapCandidateForDrawPdf no match (no overlap)");
        return null;
      }
      if (bestByIou && bestByIou.iou >= MIN_SNAP_IOU) {
        // eslint-disable-next-line no-console
        console.log("[door_detector] snapCandidateForDrawPdf chosen", { reason: "iou", id: bestByIou.id, iou: bestByIou.iou });
        return bestByIou;
      }
      if (bestByInter) {
        // eslint-disable-next-line no-console
        console.log("[door_detector] snapCandidateForDrawPdf chosen", {
          reason: "max_intersection",
          id: bestByInter.id,
          inter: bestByInter.inter,
          iou: bestByInter.iou,
        });
        return bestByInter;
      }
      if (bestByCoverage && bestByCoverage.coverage >= MIN_CAND_COVERAGE) {
        // eslint-disable-next-line no-console
        console.log("[door_detector] snapCandidateForDrawPdf chosen", {
          reason: "coverage",
          id: bestByCoverage.id,
          coverage: bestByCoverage.coverage,
          iou: bestByCoverage.iou,
        });
        return bestByCoverage;
      }
      // eslint-disable-next-line no-console
      console.log("[door_detector] snapCandidateForDrawPdf no match (overlap too weak)", {
        bestByIoU: bestByIou ? { id: bestByIou.id, iou: bestByIou.iou } : null,
      });
      return null;
    },
    [candidatePool]
  );

  // Re-render overlays and styles on prop changes.
  useEffect(() => {
    renderOverlays();
    applyDoorStyles();
  }, [applyDoorStyles, renderOverlays]);

  // Auto-focus when focusSeq changes (suppressed during edit mode).
  const focusToBBox = useCallback(
    (bbox: BBox) => {
      const root = rootRef.current;
      const ps = pageSize;
      if (!root || !ps) return;

      const cw = root.clientWidth;
      const ch = root.clientHeight;
      if (!(cw > 0) || !(ch > 0)) return;

      const [x0, y0, x1, y1] = normalizeBBox(bbox);
      const bw = Math.max(1, x1 - x0);
      const bh = Math.max(1, y1 - y0);
      const c = bboxCenter([x0, y0, x1, y1]);

      const padFactor = 3.0;
      const baseScale = baseScaleRef.current;
      const targetScale = clamp(Math.min(cw / (bw * padFactor), ch / (bh * padFactor)), baseScale, baseScale * 6);
      const targetTx = cw / 2 - targetScale * c.x;
      const targetTy = ch / 2 - targetScale * c.y;

      const start = performance.now();
      const sTx = txRef.current,
        sTy = tyRef.current,
        sScale = scaleRef.current;
      const dTx = targetTx - sTx;
      const dTy = targetTy - sTy;
      const dScale = targetScale - sScale;

      function step(now: number) {
        const t = clamp((now - start) / 260, 0, 1);
        const e = easeInOut(t);
        txRef.current = sTx + dTx * e;
        tyRef.current = sTy + dTy * e;
        scaleRef.current = sScale + dScale * e;
        applyTransform();
        if (t < 1) requestAnimationFrame(step);
        else scheduleQualityRender();
      }

      requestAnimationFrame(step);
    },
    [applyTransform, pageSize, scheduleQualityRender]
  );

  const focusToDoorId = useCallback(
    (doorId: string) => {
      const svg = svgRef.current;
      if (!svg || !doorId) return;
      const r = svg.querySelector<SVGRectElement>(`rect[data-door-id="${CSS.escape(doorId)}"]`);
      if (!r) return;
      const x = parseFloat(r.getAttribute("data-x") || r.getAttribute("x") || "0");
      const y = parseFloat(r.getAttribute("data-y") || r.getAttribute("y") || "0");
      const w = parseFloat(r.getAttribute("data-w") || r.getAttribute("width") || "0");
      const h = parseFloat(r.getAttribute("data-h") || r.getAttribute("height") || "0");
      if (!(w > 0) || !(h > 0)) return;
      focusToBBox([x, y, x + w, y + h]);
    },
    [focusToBBox]
  );

  useEffect(() => {
    focusSeqRef.current = focusSeq;
    if (editMode) return;
    if (!selectedDoorId) return;
    const lastApplied = loadState()?.focusSeq ?? null;
    if (lastApplied !== null && lastApplied === focusSeq) return;
    focusToDoorId(selectedDoorId);
  }, [editMode, focusSeq, focusToDoorId, loadState, selectedDoorId]);

  // Wheel zoom + drag pan + shift+drag drawing.
  useEffect(() => {
    const root = rootRef.current;
    const stage = stageRef.current;
    const content = contentRef.current;
    const svg = svgRef.current;
    if (!root || !stage || !content) return;

    let dragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragStartTx = 0;
    let dragStartTy = 0;

    let drawing = false;
    let drawStart: { x: number; y: number } | null = null;
    let drawRect: SVGRectElement | null = null;
    let suppressSvgClickUntil = 0;

    const clientToContent = (clientX: number, clientY: number) => {
      const cr = content.getBoundingClientRect();
      const ps = pageSize;
      if (!ps || !(cr.width > 0) || !(cr.height > 0)) return { x: 0, y: 0 };
      const qx = (clientX - cr.left) * (ps.w / cr.width);
      const qy = (clientY - cr.top) * (ps.h / cr.height);
      return { x: qx, y: qy };
    };

    const onWheel = (e: WheelEvent) => {
      if (resetBtnRef.current && e.target instanceof Node && resetBtnRef.current.contains(e.target)) return;
      e.preventDefault();
      const rect = root.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;

      const oldScale = scaleRef.current;
      const nextScale = clamp(oldScale * zoomFactor, 0.05, 20);
      const qx = (px - txRef.current) / oldScale;
      const qy = (py - tyRef.current) / oldScale;
      txRef.current = px - nextScale * qx;
      tyRef.current = py - nextScale * qy;
      scaleRef.current = nextScale;
      applyTransform();
      scheduleQualityRender();
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      if (resetBtnRef.current && e.target instanceof Node && resetBtnRef.current.contains(e.target)) return;
      if (editMode && e.shiftKey) {
        drawing = true;
        stage.style.cursor = "crosshair";
        drawStart = clientToContent(e.clientX, e.clientY);
        const temp = ensureLayer("pz_temp");
        if (temp) {
          drawRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          drawRect.setAttribute("fill", "rgb(0,255,255)");
          drawRect.setAttribute("fill-opacity", "0.12");
          drawRect.setAttribute("stroke", "rgb(0,255,255)");
          drawRect.setAttribute("stroke-opacity", "0.90");
          drawRect.setAttribute("stroke-width", "2");
          drawRect.setAttribute("stroke-dasharray", "6,4");
          drawRect.setAttribute("stroke-linecap", "square");
          drawRect.setAttribute("shape-rendering", "crispEdges");
          drawRect.setAttribute("vector-effect", "non-scaling-stroke");
          (drawRect.style as any).pointerEvents = "none";
          temp.appendChild(drawRect);
        }
        suppressSvgClickUntil = performance.now() + 250;
        e.preventDefault();
        e.stopPropagation();
        try {
          root.setPointerCapture(e.pointerId);
        } catch {
          // ignore
        }
        return;
      }

      dragging = true;
      stage.style.cursor = "grabbing";
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      dragStartTx = txRef.current;
      dragStartTy = tyRef.current;
      try {
        root.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (drawing && drawStart) {
        const p = clientToContent(e.clientX, e.clientY);
        const q = (v: number) => Math.round(v * 2) / 2;
        const x0 = q(Math.min(drawStart.x, p.x));
        const y0 = q(Math.min(drawStart.y, p.y));
        const x1 = q(Math.max(drawStart.x, p.x));
        const y1 = q(Math.max(drawStart.y, p.y));
        const w = q(Math.max(0, x1 - x0));
        const h = q(Math.max(0, y1 - y0));
        if (drawRect) {
          drawRect.setAttribute("x", String(x0));
          drawRect.setAttribute("y", String(y0));
          drawRect.setAttribute("width", String(w));
          drawRect.setAttribute("height", String(h));
        }
        e.preventDefault();
        return;
      }
      if (!dragging) return;
      txRef.current = dragStartTx + (e.clientX - dragStartX);
      tyRef.current = dragStartTy + (e.clientY - dragStartY);
      applyTransform();
    };

    const endDrag = (e: PointerEvent) => {
      if (drawing) {
        drawing = false;
        stage.style.cursor = "grab";
        const end = clientToContent(e.clientX, e.clientY);
        const x0 = Math.min(drawStart?.x ?? end.x, end.x);
        const y0 = Math.min(drawStart?.y ?? end.y, end.y);
        const x1 = Math.max(drawStart?.x ?? end.x, end.x);
        const y1 = Math.max(drawStart?.y ?? end.y, end.y);
        drawStart = null;
        if (drawRect) {
          drawRect.setAttribute("fill", "rgb(0,255,255)");
          drawRect.setAttribute("fill-opacity", "0.08");
          drawRect.setAttribute("stroke", "rgb(0,255,255)");
          drawRect.setAttribute("stroke-opacity", "0.65");
        }
        drawRect = null;

        const w = Math.abs(x1 - x0);
        const h = Math.abs(y1 - y0);
        if (w >= 2 && h >= 2) {
          const vp = viewportRef.current;
          if (!vp) return;
          const p0 = vp.convertToPdfPoint(x0, y0);
          const p1 = vp.convertToPdfPoint(x1, y1);
          const drawn: BBox = normalizeBBox([p0[0], p0[1], p1[0], p1[1]]);
          const snap = snapCandidateForDrawPdf(drawn);

          try {
            // eslint-disable-next-line no-console
            console.log("[door_detector] draw_rect endDrag (pdfjs)", {
              drawn_pdf_xyxy: drawn,
              candidates: candidatePool.length,
              snap: snap ? { id: (snap as any).id, iou: (snap as any).iou ?? computeIoU(drawn, (snap as any).bbox), bbox: (snap as any).bbox } : null,
              ts: Date.now(),
            });
          } catch {
            // ignore
          }

          // Immediate UX: draw snap/unmatched marker on the temp layer.
          const temp = ensureLayer("pz_temp");
          if (snap && temp) {
            try {
              drawBox(temp, pdfBBoxToViewportBBox(vp, snap.bbox), "rgb(0,255,0)", 3, "4,3", 0.77);
              localSelectedIdRef.current = String(snap.id || "");
              applyDoorStyles();
            } catch {
              // ignore
            }
          } else if (temp) {
            try {
              drawBox(
                temp,
                [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)],
                "rgb(255,0,255)",
                2,
                "6,4",
                0.68
              );
            } catch {
              // ignore
            }
          }

          const payload: ViewerEvent = {
            type: "draw_rect",
            event_id: randEventId(),
            bbox_pdf_xyxy: drawn,
            snapped_candidate_id: snap ? String((snap as any).id) : null,
            iou: snap && typeof (snap as any).iou === "number" ? (snap as any).iou : snap ? computeIoU(drawn, (snap as any).bbox) : null,
            snapped_bbox_pdf_xyxy: snap ? (snap as any).bbox : null,
            ts: Date.now(),
          };
          emitEvent(payload);
        }
        return;
      }

      if (!dragging) return;
      dragging = false;
      stage.style.cursor = "grab";
      scheduleQualityRender();
    };

    const onSvgPointerDownCapture = (e: PointerEvent) => {
      if (editMode && e.shiftKey) return;
      const t = e.target as any;
      if (t && t.getAttribute && t.getAttribute("data-door-id")) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const onSvgClick = (e: MouseEvent) => {
      if (performance.now && performance.now() < suppressSvgClickUntil) return;
      const t = e.target as any;
      if (!t || !t.getAttribute) return;
      const did = t.getAttribute("data-door-id");
      if (!did) return;
      e.preventDefault();
      e.stopPropagation();
      localSelectedIdRef.current = String(did);
      applyDoorStyles();
      if (!editMode) focusToDoorId(String(did));
      emitEvent({ type: "door_click", event_id: randEventId(), door_id: String(did), ts: Date.now() });
    };

    root.addEventListener("wheel", onWheel, { passive: false });
    root.addEventListener("pointerdown", onPointerDown);
    root.addEventListener("pointermove", onPointerMove);
    root.addEventListener("pointerup", endDrag);
    root.addEventListener("pointercancel", endDrag);
    root.addEventListener("pointerleave", endDrag);

    svg?.addEventListener("pointerdown", onSvgPointerDownCapture as any, true);
    svg?.addEventListener("click", onSvgClick as any);

    return () => {
      root.removeEventListener("wheel", onWheel as any);
      root.removeEventListener("pointerdown", onPointerDown as any);
      root.removeEventListener("pointermove", onPointerMove as any);
      root.removeEventListener("pointerup", endDrag as any);
      root.removeEventListener("pointercancel", endDrag as any);
      root.removeEventListener("pointerleave", endDrag as any);
      svg?.removeEventListener("pointerdown", onSvgPointerDownCapture as any, true);
      svg?.removeEventListener("click", onSvgClick as any);
    };
  }, [applyDoorStyles, applyTransform, drawBox, editMode, emitEvent, ensureLayer, focusToDoorId, pageSize, scheduleQualityRender, snapCandidateForDrawPdf]);

  // React to prop changes like the old pollSelection loop.
  useEffect(() => {
    const did = selectedDoorId || null;
    const needsStyle =
      did !== lastSelectedIdRef.current ||
      viewerDisplayMode !== lastViewerDisplayRef.current ||
      editMode !== lastEditModeRef.current;
    if (did !== lastSelectedIdRef.current) {
      lastSelectedIdRef.current = did;
      localSelectedIdRef.current = null;
    }
    lastViewerDisplayRef.current = viewerDisplayMode;
    if (editMode !== lastEditModeRef.current) {
      const wasEditing = !!lastEditModeRef.current;
      lastEditModeRef.current = editMode;
      if (wasEditing && !editMode) {
        const manual = ensureLayer("pz_manual");
        const temp = ensureLayer("pz_temp");
        clearSvgLayer(manual);
        clearSvgLayer(temp);
      }
    }
    if (needsStyle) applyDoorStyles();
  }, [applyDoorStyles, clearSvgLayer, editMode, ensureLayer, selectedDoorId, viewerDisplayMode]);

  return (
    <div
      ref={rootRef}
      style={{
        width: "100%",
        height: `${height}px`,
        overflow: "hidden",
        background: "#0e1117",
        borderRadius: 6,
        border: "1px solid rgba(255,255,255,0.12)",
        position: "relative",
      }}
    >
      <button
        ref={resetBtnRef}
        type="button"
        aria-label="Reset view"
        aria-hidden={!resetVisible}
        tabIndex={resetVisible ? 0 : -1}
        onPointerDownCapture={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          resetView();
        }}
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          zIndex: 5,
          padding: "6px 10px",
          borderRadius: 999,
          border: "1px solid rgba(255, 255, 255, 0.16)",
          background: "rgba(17, 25, 40, 0.65)",
          color: "rgba(255, 255, 255, 0.92)",
          fontSize: 12,
          lineHeight: 1,
          cursor: "pointer",
          backdropFilter: "blur(6px)",
          WebkitBackdropFilter: "blur(6px)",
          boxShadow: "0 6px 22px rgba(0, 0, 0, 0.35)",
          opacity: resetVisible ? 1 : 0,
          transform: resetVisible ? "translateY(0px)" : "translateY(-4px)",
          transition:
            "opacity 120ms ease, transform 120ms ease, border-color 120ms ease, background 120ms ease",
          userSelect: "none",
          pointerEvents: resetVisible ? "auto" : "none",
        }}
      >
        Reset
      </button>
      <div
        ref={stageRef}
        style={{
          width: "100%",
          height: "100%",
          position: "relative",
          userSelect: "none",
          cursor: "grab",
          touchAction: "none",
        }}
      >
        <div
          ref={contentRef}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: pageSize ? `${pageSize.w}px` : "1px",
            height: pageSize ? `${pageSize.h}px` : "1px",
            transformOrigin: "0 0",
            willChange: "transform",
          }}
        >
          <canvas
            ref={canvasARef}
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              pointerEvents: "none",
              opacity: activeCanvas === "a" ? 1 : 0,
              visibility: activeCanvas === "a" ? "visible" : "hidden",
            }}
          />
          <canvas
            ref={canvasBRef}
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              pointerEvents: "none",
              opacity: activeCanvas === "b" ? 1 : 0,
              visibility: activeCanvas === "b" ? "visible" : "hidden",
            }}
          />
          <svg
            ref={svgRef}
            width={pageSize?.w ?? 1}
            height={pageSize?.h ?? 1}
            viewBox={`0 0 ${pageSize?.w ?? 1} ${pageSize?.h ?? 1}`}
            style={{ position: "absolute", left: 0, top: 0, pointerEvents: "auto" }}
            xmlns="http://www.w3.org/2000/svg"
          />
        </div>
      </div>
    </div>
  );
}

