/**
 * Render-ahead frame scheduling, apart from the DOM and the GPU so that it
 * runs in a simulation. Frames are rendered as soon as they arrive into a
 * ring of slots (one ImageBitmap or texture each, say), and each animation
 * frame presents the one that is due.
 */

/** A frame waiting to be rendered. */
export interface PacedFrame {
  /** Media time of the frame, in ms: when its audio plays. */
  readonly mediaMs: number;
  /** Frames of a generation other than the current one (an earlier avatar, say) are skipped. */
  readonly generation: number;
}

export interface PacerJob<F extends PacedFrame> {
  readonly frame: F;
  /** The ring slot the frame renders into. */
  readonly slot: number;
}

export interface PacerCounts {
  rendered: number;
  renderedAhead: number;
  presented: number;
  /** Arrived or rendered after a later frame was already shown. */
  skippedLate: number;
  skippedOtherGeneration: number;
  /** Rendered but overtaken by a later due frame before it was shown. */
  skippedOverdue: number;
  /** Dropped while waiting because the ring was full. */
  droppedRingFull: number;
  /** Rendered frames sent back to the queue to make room for an earlier late arrival. */
  requeuedRingFull: number;
  queueOverflow: number;
}

export interface PacerOptions {
  /** Slots in the ring, the most frames rendered ahead. */
  slots?: number;
  /** Frame period in ms. */
  periodMs?: number;
  /** Most frames waiting to be rendered; the oldest go first beyond it. */
  maxQueue?: number;
}

export interface Pacer<F extends PacedFrame> {
  readonly counts: PacerCounts;
  add(frame: F): void;
  /** The next frame to render and its slot, or null when nothing is to render or no slot is free. */
  takeNext(nowMs: number, dueAt: (f: F) => number, generation: number): PacerJob<F> | null;
  /** Reports a job rendered into its slot. */
  finish(job: PacerJob<F>, nowMs: number, dueAt: (f: F) => number): void;
  /** Gives back the slot of a job that will not finish. */
  cancel(job: PacerJob<F>): void;
  /** The job to show at `nowMs`, or null to keep showing the current one. */
  present(nowMs: number, dueAt: (f: F) => number): PacerJob<F> | null;
  /** Drops every frame not yet shown and returns how many. */
  clear(): number;
  /** For a new set of slots: rendered frames go back to be rendered again. */
  reset(): void;
}

function insertByMedia<T>(list: T[], item: T, media: (x: T) => number) {
  let i = list.length;
  while (i > 0 && media(list[i - 1]) > media(item)) i--;
  list.splice(i, 0, item);
}

export function createPacer<F extends PacedFrame>({ slots = 8, periodMs = 40, maxQueue = 250 }: PacerOptions = {}): Pacer<F> {
  const counts: PacerCounts = {
    rendered: 0, renderedAhead: 0, presented: 0, skippedLate: 0, skippedOtherGeneration: 0,
    skippedOverdue: 0, droppedRingFull: 0, requeuedRingFull: 0, queueOverflow: 0,
  };
  const queue: F[] = [];
  const ring: PacerJob<F>[] = [];
  let free = [...Array(slots).keys()];
  let shownSlot: number | null = null, lastShownMedia = -Infinity, ringWasFull = false;
  const releaseSlot = (slot: number) => { if (slot !== shownSlot) free.push(slot); };

  return {
    counts,
    add(frame) {
      if (frame.mediaMs <= lastShownMedia) { counts.skippedLate++; return; }
      insertByMedia(queue, frame, (x) => x.mediaMs);
      if (queue.length > maxQueue) { queue.shift(); counts.queueOverflow++; }
    },

    takeNext(now, dueAt, generation) {
      while (queue.length) {
        const head = queue[0];
        if (head.generation !== generation) { queue.shift(); counts.skippedOtherGeneration++; continue; }
        // A newer frame is queued behind it, so rendering this one would
        // only delay that one.
        if (queue.length > 1 && dueAt(head) < now - periodMs) {
          queue.shift();
          if (ringWasFull) counts.droppedRingFull++; else counts.skippedLate++;
          continue;
        }
        if (!free.length) {
          // A full ring holds the frames due soonest, so rendering further
          // ahead can wait. Dropping one instead would lose a frame that is
          // on time, since frames often arrive further ahead than the ring
          // reaches. Only a late arrival due before the ring's last frame
          // takes that frame's slot, and that frame is rendered again later.
          const last = ring.at(-1);
          if (last && last.frame.mediaMs > head.mediaMs) {
            ring.pop();
            free.push(last.slot);
            insertByMedia(queue, last.frame, (x) => x.mediaMs);
            counts.requeuedRingFull++;
            continue;
          }
          ringWasFull = true;
          return null;
        }
        ringWasFull = false;
        queue.shift();
        return { frame: head, slot: free.pop()! };
      }
      return null;
    },

    finish(job, now, dueAt) {
      counts.rendered++;
      if (job.frame.mediaMs <= lastShownMedia) { counts.skippedLate++; releaseSlot(job.slot); return; }
      if (dueAt(job.frame) > now) counts.renderedAhead++;
      insertByMedia(ring, job, (x) => x.frame.mediaMs);
    },

    cancel(job) { releaseSlot(job.slot); },

    present(now, dueAt) {
      while (ring.length > 1 && dueAt(ring[0].frame) < now - periodMs && dueAt(ring[1].frame) <= now) {
        releaseSlot(ring.shift()!.slot);
        counts.skippedOverdue++;
      }
      if (!ring.length || dueAt(ring[0].frame) > now) return null;
      const job = ring.shift()!;
      if (shownSlot !== null) free.push(shownSlot);
      shownSlot = job.slot;
      lastShownMedia = job.frame.mediaMs;
      counts.presented++;
      return job;
    },

    clear() {
      const n = queue.length + ring.length;
      queue.length = 0;
      for (const job of ring.splice(0)) releaseSlot(job.slot);
      return n;
    },

    reset() {
      for (const job of ring.splice(0)) insertByMedia(queue, job.frame, (x) => x.mediaMs);
      free = [...Array(slots).keys()];
      shownSlot = null;
      ringWasFull = false;
    },
  };
}
