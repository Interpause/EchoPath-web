/**
 * 3D spatial audio engine using Web Audio API.
 * Coordinate system:
 *   X: negative = left,    positive = right
 *   Y: negative = below,   positive = above
 *   Z: negative = in front, positive = behind
 */

export interface SpatialAudioOptions {
  /** Base oscillator frequency in Hz. Default: 440 */
  baseFrequency?: number;
  /** Beep duration in seconds. Default: 0.2 */
  duration?: number;
  /** Distance at which volume reaches 0. Default: 30 */
  maxDistance?: number;
  /** Oscillator wave shape. Default: 'sine' */
  waveType?: OscillatorType;
  /** Slightly lower pitch for distant objects. Default: true */
  shiftFreqWithDistance?: boolean;
}

export interface AudioStats {
  distance: number;
  /** 0 (silent) to 1 (full volume) */
  volume: number;
  /** Human-readable dominant direction, e.g. 'left', 'above', 'in front' */
  direction: string;
  /** Stereo pan value: -1 = hard left, 0 = center, +1 = hard right */
  pan: number;
}

export class SpatialAudioEngine {
  private ctx: AudioContext;
  private options: Required<SpatialAudioOptions>;

  constructor(options: SpatialAudioOptions = {}) {
    this.ctx = new AudioContext();
    this.options = {
      baseFrequency:          options.baseFrequency          ?? 440,
      duration:               options.duration               ?? 0.2,
      maxDistance:            options.maxDistance            ?? 30,
      waveType:               options.waveType               ?? 'sine',
      shiftFreqWithDistance:  options.shiftFreqWithDistance  ?? true,
    };
    this._setupListener();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Play a single beep positioned at (x, y, z).
   * Silently skips if the object is beyond maxDistance.
   */
  async playBeep(
    x: number,
    y: number,
    z: number,
    overrides: Partial<SpatialAudioOptions> = {}
  ): Promise<void> {
    if (this.ctx.state === 'suspended') await this.ctx.resume();

    const opts = { ...this.options, ...overrides };
    const stats = this.computeStats(x, y, z);

    if (stats.volume <= 0) return;

    const now = this.ctx.currentTime;
    const dur = opts.duration;

    // Oscillator — slightly lower pitch when far away
    const osc = this.ctx.createOscillator();
    osc.type = opts.waveType;
    const freqMult = opts.shiftFreqWithDistance
      ? 1 - (stats.distance / opts.maxDistance) * 0.15
      : 1;
    osc.frequency.value = opts.baseFrequency * freqMult;

    // Gain — smooth attack + release envelope, amplitude = distance-based volume
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(stats.volume, now + 0.01);
    gain.gain.linearRampToValueAtTime(0, now + dur);

    // StereoPannerNode — deterministic L/R positioning from azimuth angle.
    //
    // atan2(x, -z) gives the horizontal angle:
    //   directly left  (x=-1, z=0)  → -π/2 → pan = -1.0
    //   directly right (x=+1, z=0)  → +π/2 → pan = +1.0
    //   directly front (x=0,  z=-1) →  0   → pan =  0.0
    //   directly behind(x=0,  z=+1) →  π   → pan =  0.0  (behind = centered, correct)
    //
    // Clamped to [-1, 1] so diagonal positions don't exceed bounds.
    const stereoPanner = this.ctx.createStereoPanner();
    stereoPanner.pan.setValueAtTime(stats.pan, now);

    // PannerNode (HRTF) — retained only for Y-axis elevation cues that
    // StereoPannerNode cannot model. Volume rolloff is disabled here (rolloffFactor=0)
    // because gain is handled manually above.
    const panner = this.ctx.createPanner();
    panner.panningModel  = 'HRTF';
    panner.distanceModel = 'linear';
    panner.refDistance   = 1;
    panner.maxDistance   = opts.maxDistance;
    panner.rolloffFactor = 0; // gain node owns volume — no double rolloff

    if (panner.positionX) {
      panner.positionX.setValueAtTime(x, now);
      panner.positionY.setValueAtTime(y, now);
      panner.positionZ.setValueAtTime(z, now);
    } else {
      // Safari / older browser fallback
      (panner as any).setPosition(x, y, z);
    }

    // Signal chain: osc → gain → stereoPanner → panner → speakers
    osc.connect(gain);
    gain.connect(stereoPanner);
    stereoPanner.connect(panner);
    panner.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + dur + 0.05);
  }

  /**
   * Play repeating pulses at (x, y, z).
   * Pulse interval is shorter when the object is closer (like a parking sensor).
   *   distance=0   → ~300 ms interval
   *   distance=max → ~2000 ms interval
   *
   * Returns a stop() function — call it to cancel the pulse.
   */
  startPulse(
    x: number,
    y: number,
    z: number,
    overrides: Partial<SpatialAudioOptions> = {}
  ): () => void {
    const { distance } = this.computeStats(x, y, z);
    const intervalMs = Math.max(200, Math.min(2000, 300 + distance * 55));

    this.playBeep(x, y, z, overrides);
    const id = setInterval(() => this.playBeep(x, y, z, overrides), intervalMs);
    return () => clearInterval(id);
  }

  /** Close and release audio resources. */
  async dispose(): Promise<void> {
    if (this.ctx.state !== "closed") {
      await this.ctx.close();
    }
  }

  /**
   * Compute stats for a position without playing any audio.
   * Useful for UI display or debugging.
   */
  computeStats(x: number, y: number, z: number): AudioStats {
    const distance = Math.sqrt(x * x + y * y + z * z);
    const volume   = Math.max(0, Math.min(1, 1 - distance / this.options.maxDistance));

    // Dominant direction label
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    let direction: string;
    if (ax >= ay && ax >= az)      direction = x < 0 ? 'left'     : 'right';
    else if (ay >= ax && ay >= az) direction = y > 0 ? 'above'    : 'below';
    else                           direction = z < 0 ? 'in front' : 'behind';

    // Stereo pan: atan2(x, -z) → normalize to [-1, 1] via π/2
    const azimuth = Math.atan2(x, -z);
    const pan = Math.max(-1, Math.min(1, azimuth / (Math.PI / 2)));

    return { distance, volume, direction, pan };
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private _setupListener(): void {
    const { listener } = this.ctx;
    // Listener at origin, facing forward (negative Z), head upright (positive Y)
    if (listener.positionX) {
      listener.positionX.value =  0;
      listener.positionY.value =  0;
      listener.positionZ.value =  0;
      listener.forwardX.value  =  0;
      listener.forwardY.value  =  0;
      listener.forwardZ.value  = -1;
      listener.upX.value       =  0;
      listener.upY.value       =  1;
      listener.upZ.value       =  0;
    } else {
      // Safari fallback
      (listener as any).setPosition(0, 0, 0);
      (listener as any).setOrientation(0, 0, -1, 0, 1, 0);
    }
  }
}