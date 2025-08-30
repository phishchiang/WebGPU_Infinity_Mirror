import { mat4, vec3, Vec3, Mat4 } from 'wgpu-matrix';
import { CameraBase } from './camera';
import Input from './input';

/**
 * Pose data produced by a face/head tracker.
 * All angles in radians. distance is the desired camera distance from target.
 */
export interface HeadPose {
  yaw: number;      // left/right rotation (positive yaw turns to the left looking from above)
  pitch: number;    // up/down rotation (positive looks up)
  distance: number; // target distance from origin (>0)
  confidence?: number; // optional confidence 0..1
}

/** Options for HeadTrackedCamera */
export interface HeadTrackedCameraOptions {
  /** Initial yaw */
  yaw?: number;
  /** Initial pitch */
  pitch?: number;
  /** Initial distance */
  distance?: number;
  /** Smoothing half-life (seconds) for rotation; smaller = snappier */
  rotationHalfLife?: number;
  /** Smoothing half-life (seconds) for distance */
  distanceHalfLife?: number;
  /** Clamp limits */
  maxYaw?: number;   // radians, symmetric ±
  maxPitch?: number; // radians, symmetric ±
  minDistance?: number;
  maxDistance?: number;
}

/** Utility to compute exponential smoothing factor for a half-life */
function smoothingAlpha(dt: number, halfLife: number): number {
  // alpha = 1 - 0.5^(dt/halfLife)
  return halfLife <= 0 ? 1 : 1 - Math.pow(0.5, dt / halfLife);
}

/** Clamp helper */
function clamp(x: number, a: number, b: number) { return Math.min(Math.max(x, a), b); }

/**
 * HeadTrackedCamera consumes target head pose each frame and eases camera orientation & distance.
 * The camera always looks at the world origin (0,0,0) with back vector pointing from origin to camera.
 */
export class HeadTrackedCamera extends CameraBase {
  private yaw: number;
  private pitch: number;
  private distance: number;

  private targetYaw: number;
  private targetPitch: number;
  private targetDistance: number;

  private readonly rotationHalfLife: number;
  private readonly distanceHalfLife: number;

  private readonly maxYaw: number;
  private readonly maxPitch: number;
  private readonly minDistance: number;
  private readonly maxDistance: number;

  // Last supplied pose time or confidence if needed (not used yet)
  private poseConfidence: number = 1;

  constructor(opts: HeadTrackedCameraOptions = {}) {
    super();
    this.yaw = this.targetYaw = opts.yaw ?? 0;
    this.pitch = this.targetPitch = opts.pitch ?? 0;
    this.distance = this.targetDistance = opts.distance ?? 6;

    this.rotationHalfLife = opts.rotationHalfLife ?? 0.12; // ~120ms smoothing
    this.distanceHalfLife = opts.distanceHalfLife ?? 0.18;

    this.maxYaw = opts.maxYaw ?? 0.6; // ~34°
    this.maxPitch = opts.maxPitch ?? 0.4; // ~23°
    this.minDistance = opts.minDistance ?? 2.0;
    this.maxDistance = opts.maxDistance ?? 15.0;

    this.updateMatrix();
  }

  /** Supply a new target pose (raw). Values get clamped & stored as targets. */
  setPose(pose: HeadPose) {
    this.targetYaw = clamp(pose.yaw, -this.maxYaw, this.maxYaw);
    this.targetPitch = clamp(pose.pitch, -this.maxPitch, this.maxPitch);
    this.targetDistance = clamp(pose.distance, this.minDistance, this.maxDistance);
    if (pose.confidence != null) this.poseConfidence = pose.confidence; // future weighting
  }

  /** Update the camera each frame. Input deltas ignored (head tracking drives it). */
  update(deltaTime: number, _input: Input): Mat4 {
    // Exponential smoothing toward targets
    const aRot = smoothingAlpha(deltaTime, this.rotationHalfLife);
    this.yaw += (this.targetYaw - this.yaw) * aRot;
    this.pitch += (this.targetPitch - this.pitch) * aRot;

    const aDist = smoothingAlpha(deltaTime, this.distanceHalfLife);
    this.distance += (this.targetDistance - this.distance) * aDist;

    this.updateMatrix();
    this.view = mat4.invert(this.matrix);
    return this.view;
  }

  /** Build camera basis from yaw/pitch/distance (looking at origin). */
  private updateMatrix() {
    // Back vector points from camera toward origin (camera looks forward = -back)
    // Start with spherical coords using yaw around Y and pitch around X (applied after yaw)
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // Back direction = normalized camera position vector (pointing from origin to camera)
    const back = vec3.normalize(vec3.create(Math.sin(this.yaw) * cp, -sp, Math.cos(this.yaw) * cp));
    // Position = back * distance
    const pos = vec3.scale(back, this.distance);

    // Choose global up as reference to build right/up
    const worldUp = vec3.create(0, 1, 0);
    let right = vec3.cross(worldUp, back);
    if (vec3.len(right) < 1e-5) {
      // Gimbal-ish case; use fallback axis
      right = vec3.create(1, 0, 0);
    } else {
      right = vec3.normalize(right);
    }
    const up = vec3.normalize(vec3.cross(back, right));

    // Write into base matrix columns (right, up, back, pos)
    this.right = right;
    this.up = up;
    this.back = back;
    this.position = pos;
  }
}
