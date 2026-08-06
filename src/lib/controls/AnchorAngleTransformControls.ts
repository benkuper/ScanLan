import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';

type FixedRotationAxis = 'X' | 'Y' | 'Z' | 'E';

interface AngleDrag {
  accumulatedAngle: number;
  center: THREE.Vector3;
  orientation: number;
  parentWorldQuaternionInverse: THREE.Quaternion;
  previousPointerAngle: number | null;
  startQuaternion: THREE.Quaternion;
  worldAxis: THREE.Vector3;
}

const AXIS_VECTORS: Record<Exclude<FixedRotationAxis, 'E'>, THREE.Vector3> = {
  X: new THREE.Vector3(1, 0, 0),
  Y: new THREE.Vector3(0, 1, 0),
  Z: new THREE.Vector3(0, 0, 1)
};
const MIN_POINTER_RADIUS_PX = 6;

function isFixedRotationAxis(axis: TransformControls['axis']): axis is FixedRotationAxis {
  return axis === 'X' || axis === 'Y' || axis === 'Z' || axis === 'E';
}

function shortestAngleDelta(from: number, to: number): number {
  let delta = to - from;
  if (delta > Math.PI) delta -= Math.PI * 2;
  else if (delta < -Math.PI) delta += Math.PI * 2;
  return delta;
}

/**
 * Uses the mouse's polar angle around the projected transform origin for fixed-axis
 * rotation. Unlike Three.js's default linear drag speed, this makes a given mouse
 * movement less sensitive as the pointer moves farther away from the gizmo.
 */
export class AnchorAngleTransformControls extends TransformControls {
  private angleDrag: AngleDrag | null = null;

  override pointerDown(pointer: PointerEvent | null): void {
    super.pointerDown(pointer);
    this.angleDrag = null;

    const axis = this.axis;
    if (
      pointer === null
      || !this.dragging
      || this.mode !== 'rotate'
      || !isFixedRotationAxis(axis)
      || !this.object
    ) return;

    this.object.updateWorldMatrix(true, false);
    this.camera.updateWorldMatrix(true, false);
    const center = this.object.getWorldPosition(new THREE.Vector3());
    const eye = this.camera.getWorldPosition(new THREE.Vector3()).sub(center).normalize();
    const worldAxis = axis === 'E'
      ? eye.clone()
      : AXIS_VECTORS[axis].clone();

    if (axis !== 'E' && this.space === 'local') {
      worldAxis.applyQuaternion(this.object.getWorldQuaternion(new THREE.Quaternion())).normalize();
    }

    const parentWorldQuaternionInverse = this.object.parent
      ? this.object.parent.getWorldQuaternion(new THREE.Quaternion()).invert()
      : new THREE.Quaternion();
    const pointerAngle = this.pointerAngle(pointer, center);
    if (pointerAngle === null) return;

    this.angleDrag = {
      accumulatedAngle: 0,
      center,
      orientation: worldAxis.dot(eye) < 0 ? -1 : 1,
      parentWorldQuaternionInverse,
      previousPointerAngle: pointerAngle,
      startQuaternion: this.object.quaternion.clone(),
      worldAxis
    };
  }

  override pointerMove(pointer: PointerEvent | null): void {
    const drag = this.angleDrag;
    if (
      drag === null
      || pointer === null
      || this.mode !== 'rotate'
      || !isFixedRotationAxis(this.axis)
      || !this.dragging
    ) {
      super.pointerMove(pointer);
      return;
    }
    if (pointer.button !== -1) return;

    const pointerAngle = this.pointerAngle(pointer, drag.center);
    if (pointerAngle === null) {
      // The angular direction is undefined at the anchor. Resume from the other
      // side without injecting a sudden half-turn if the pointer crosses it.
      drag.previousPointerAngle = null;
      return;
    }
    if (drag.previousPointerAngle === null) {
      drag.previousPointerAngle = pointerAngle;
      return;
    }

    drag.accumulatedAngle += shortestAngleDelta(drag.previousPointerAngle, pointerAngle) * drag.orientation;
    drag.previousPointerAngle = pointerAngle;
    const rotationAngle = this.rotationSnap
      ? Math.round(drag.accumulatedAngle / this.rotationSnap) * this.rotationSnap
      : drag.accumulatedAngle;
    const axisInParentSpace = drag.worldAxis.clone().applyQuaternion(drag.parentWorldQuaternionInverse);
    this.object.quaternion
      .setFromAxisAngle(axisInParentSpace, rotationAngle)
      .multiply(drag.startQuaternion)
      .normalize();

    const rotationState = this as unknown as {
      rotationAngle: number;
      rotationAxis: THREE.Vector3;
    };
    rotationState.rotationAngle = rotationAngle;
    rotationState.rotationAxis.copy(axisInParentSpace);
    this.dispatchEvent({ type: 'change' });
    this.dispatchEvent({ type: 'objectChange' });
  }

  override pointerUp(pointer: PointerEvent | null): void {
    super.pointerUp(pointer);
    if (!this.dragging) this.angleDrag = null;
  }

  private pointerAngle(pointer: PointerEvent, center: THREE.Vector3): number | null {
    if (!this.domElement) return null;
    const bounds = this.domElement.getBoundingClientRect();
    const projectedCenter = center.clone().project(this.camera);
    const dx = (pointer.x - projectedCenter.x) * bounds.width * 0.5;
    const dy = (pointer.y - projectedCenter.y) * bounds.height * 0.5;
    if (Math.hypot(dx, dy) < MIN_POINTER_RADIUS_PX) return null;
    return Math.atan2(dy, dx);
  }
}
