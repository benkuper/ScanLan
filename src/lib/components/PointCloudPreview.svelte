<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { TransformControls, type TransformControlsMode } from 'three/examples/jsm/controls/TransformControls.js';
  import type { CameraFrame, CloudTransform, PackedPreviewFrame, PreviewMesh, PreviewPoint } from '../types';

  export let points: PreviewPoint[] = [];
  export let packedFrame: PackedPreviewFrame | null = null;
  export let processing = false;
  export let live = false;
  export let pointSize = 0.034;
  export let opacity = 0.92;
  export let showColors = true;
  export let renderMode: 'points' | 'mesh' = 'points';
  export let mesh: PreviewMesh | null = null;
  export let cameraFrames: CameraFrame[] = [];
  export let showCameraFrames = false;
  export let floorPickMode = false;
  export let anchorPickMode = false;
  export let cloudTransform: CloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
  export let gizmoAnchor: [number, number, number] = [0, 0, 0];
  export let editMode = false;
  export let gizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
  export let onFloorDetected: (transform: CloudTransform) => void = () => undefined;
  export let onFloorMessage: (message: string) => void = () => undefined;
  export let onAnchorPicked: (anchor: [number, number, number]) => void = () => undefined;
  export let onTransformChanged: (transform: CloudTransform) => void = () => undefined;
  export let onTransformCommitted: () => void = () => undefined;

  let canvas: HTMLCanvasElement;
  let setGeometry: (nextPoints: PreviewPoint[]) => void = () => undefined;
  let setPackedGeometry: (frame: PackedPreviewFrame) => void = () => undefined;
  let setMaterial: (size: number, alpha: number, colors: boolean) => void = () => undefined;
  let setMesh: (nextMesh: PreviewMesh | null) => void = () => undefined;
  let setRenderMode: (mode: 'points' | 'mesh') => void = () => undefined;
  let setTransform: (transform: CloudTransform, anchor: [number, number, number]) => void = () => undefined;
  let setGizmo: (enabled: boolean, mode: 'translate' | 'rotate' | 'scale') => void = () => undefined;
  let setCameraFrames: (frames: CameraFrame[], visible: boolean) => void = () => undefined;

  $: {
    if (packedFrame) setPackedGeometry(packedFrame);
    else setGeometry(points);
  }
  $: setMaterial(pointSize, opacity, showColors);
  $: setMesh(mesh);
  $: setRenderMode(renderMode);
  $: setTransform(cloudTransform, gizmoAnchor);
  $: setGizmo(editMode, gizmoMode);
  $: setCameraFrames(cameraFrames, showCameraFrames);

  onMount(() => {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07111c');
    scene.fog = new THREE.FogExp2('#07111c', 0.055);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    camera.position.set(6.8, 4.7, 7.6);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.target.set(0, 1.1, 0);
    controls.minDistance = 0.5;
    controls.maxDistance = 30;

    const grid = new THREE.GridHelper(12, 24, '#19384a', '#102a39');
    grid.position.y = -0.015;
    scene.add(grid);

    // The transform control is attached to this pivot. Offsetting the points by
    // the inverse anchor keeps the cloud's persisted transform equivalent to
    // T(position) * R(rotation) * S(scale), regardless of where the gizmo sits.
    const pivotGroup = new THREE.Group();
    const anchor = new THREE.Vector3();
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: showColors,
      color: '#a9dce8',
      sizeAttenuation: true,
      transparent: true,
      depthWrite: opacity >= 0.98,
      opacity
    });
    const cloud = new THREE.Points(geometry, material);
    pivotGroup.add(cloud);
    const meshGroup = new THREE.Group();
    const meshMaterial = new THREE.MeshBasicMaterial({
      color: '#a9dce8',
      side: THREE.DoubleSide,
      transparent: opacity < 1,
      // The mesh is assembled from overlapping RGB-D keyframe patches. Keep
      // depth writes on so rear patches do not show through the nearest one.
      depthWrite: true,
      opacity
    });
    pivotGroup.add(meshGroup);
    const frustumGeometry = new THREE.BufferGeometry();
    const frustumMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.82,
      depthWrite: false
    });
    const frustums = new THREE.LineSegments(frustumGeometry, frustumMaterial);
    frustums.renderOrder = 3;
    pivotGroup.add(frustums);
    scene.add(pivotGroup);

    const transformControls = new TransformControls(camera, renderer.domElement);
    const transformHelper = transformControls.getHelper();
    transformControls.setSpace('world');
    transformControls.setSize(0.82);
    transformControls.translationSnap = 0.01;
    transformControls.rotationSnap = THREE.MathUtils.degToRad(1);
    transformControls.scaleSnap = 0.01;
    transformHelper.visible = false;
    scene.add(transformHelper);

    setGeometry = (nextPoints) => {
      const positions = new Float32Array(nextPoints.length * 3);
      const colors = new Float32Array(nextPoints.length * 3);
      nextPoints.forEach((point, index) => {
        positions.set(point.position, index * 3);
        colors.set(point.color.map((channel) => channel / 255), index * 3);
      });
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.computeBoundingSphere();
    };
    setPackedGeometry = (frame) => {
      geometry.setAttribute('position', new THREE.BufferAttribute(frame.positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(frame.colors, 3, true));
      geometry.computeBoundingSphere();
    };
    let loadedMeshTexture: THREE.Texture | null = null;
    let pendingTextureUrl: string | null = null;
    let meshLoadGeneration = 0;
    let meshColorsEnabled = showColors;
    let activeRenderMode: 'points' | 'mesh' = renderMode;
    const clearMesh = () => {
      meshGroup.traverse((object) => {
        if (object instanceof THREE.Mesh) object.geometry.dispose();
      });
      meshGroup.clear();
      loadedMeshTexture?.dispose();
      loadedMeshTexture = null;
      meshMaterial.map = null;
      meshMaterial.needsUpdate = true;
      if (pendingTextureUrl) URL.revokeObjectURL(pendingTextureUrl);
      pendingTextureUrl = null;
    };
    setMesh = (nextMesh) => {
      meshLoadGeneration += 1;
      const generation = meshLoadGeneration;
      clearMesh();
      if (!nextMesh) return;

      const previewGeometry = new THREE.BufferGeometry();
      previewGeometry.setAttribute('position', new THREE.BufferAttribute(nextMesh.positions, 3));
      previewGeometry.setAttribute('uv', new THREE.BufferAttribute(nextMesh.uvs, 2));
      previewGeometry.setIndex(new THREE.BufferAttribute(nextMesh.indices, 1));
      previewGeometry.computeBoundingSphere();
      meshGroup.add(new THREE.Mesh(previewGeometry, meshMaterial));

      const textureBytes = Uint8Array.from(nextMesh.texture);
      pendingTextureUrl = URL.createObjectURL(new Blob([textureBytes.buffer], { type: 'image/png' }));
      const textureUrl = pendingTextureUrl;
      new THREE.TextureLoader().load(
        textureUrl,
        (texture) => {
          URL.revokeObjectURL(textureUrl);
          if (pendingTextureUrl === textureUrl) pendingTextureUrl = null;
          if (generation !== meshLoadGeneration) {
            texture.dispose();
            return;
          }
          texture.colorSpace = THREE.SRGBColorSpace;
          loadedMeshTexture = texture;
          meshMaterial.map = meshColorsEnabled ? texture : null;
          meshMaterial.needsUpdate = true;
        },
        undefined,
        () => {
          URL.revokeObjectURL(textureUrl);
          if (pendingTextureUrl === textureUrl) pendingTextureUrl = null;
        }
      );
    };
    setMaterial = (size, alpha, colors) => {
      material.size = size;
      material.opacity = alpha;
      material.depthWrite = alpha >= 0.98;
      material.vertexColors = colors;
      material.needsUpdate = true;
      meshColorsEnabled = colors;
      meshMaterial.opacity = alpha;
      meshMaterial.transparent = alpha < 1;
      meshMaterial.depthWrite = true;
      meshMaterial.map = colors ? loadedMeshTexture : null;
      meshMaterial.color.set(colors ? '#ffffff' : '#a9dce8');
      meshMaterial.needsUpdate = true;
    };
    setRenderMode = (mode) => {
      activeRenderMode = mode;
      cloud.visible = mode === 'points';
      meshGroup.visible = mode === 'mesh';
    };
    setCameraFrames = (frames, visible) => {
      frustums.visible = visible && frames.length > 0;
      if (!frustums.visible) return;
      const sampled = frames.length <= 240
        ? frames
        : Array.from({ length: 240 }, (_, index) => frames[Math.round(index * (frames.length - 1) / 239)]);
      const positions: number[] = [];
      const colors: number[] = [];
      const previousByPhase = new Map<string, THREE.Vector3>();
      for (const frame of sampled) {
        const matrix = new THREE.Matrix4().set(...frame.matrix);
        const depth = frame.textureFrame ? 0.24 : 0.17;
        const halfHeight = Math.tan(THREE.MathUtils.degToRad(frame.fovYDegrees) / 2) * depth;
        const halfWidth = halfHeight * frame.aspect;
        const ySign = frame.imageYUp ? 1 : -1;
        const origin = new THREE.Vector3(0, 0, 0).applyMatrix4(matrix);
        const corners = [
          new THREE.Vector3(-halfWidth, ySign * halfHeight, depth),
          new THREE.Vector3(halfWidth, ySign * halfHeight, depth),
          new THREE.Vector3(halfWidth, -ySign * halfHeight, depth),
          new THREE.Vector3(-halfWidth, -ySign * halfHeight, depth)
        ].map((corner) => corner.applyMatrix4(matrix));
        let hash = 0;
        for (const character of frame.phaseId) hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
        const color = new THREE.Color().setHSL(((Math.abs(hash) % 280) + 25) / 360, 0.72, frame.textureFrame ? 0.67 : 0.52);
        const addSegment = (start: THREE.Vector3, end: THREE.Vector3, brightness = 1) => {
          positions.push(start.x, start.y, start.z, end.x, end.y, end.z);
          colors.push(
            color.r * brightness, color.g * brightness, color.b * brightness,
            color.r * brightness, color.g * brightness, color.b * brightness
          );
        };
        corners.forEach((corner) => addSegment(origin, corner));
        for (let index = 0; index < 4; index += 1) addSegment(corners[index], corners[(index + 1) % 4]);
        const previous = previousByPhase.get(frame.phaseId);
        if (previous) addSegment(previous, origin, 0.48);
        previousByPhase.set(frame.phaseId, origin);
      }
      frustumGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      frustumGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      frustumGeometry.computeBoundingSphere();
    };
    setTransform = (transform, nextAnchor) => {
      anchor.fromArray(nextAnchor);
      cloud.position.copy(anchor).multiplyScalar(-1);
      meshGroup.position.copy(cloud.position);
      frustums.position.copy(cloud.position);
      pivotGroup.rotation.set(
        THREE.MathUtils.degToRad(transform.rotation[0]),
        THREE.MathUtils.degToRad(transform.rotation[1]),
        THREE.MathUtils.degToRad(transform.rotation[2]),
        'XYZ'
      );
      pivotGroup.scale.fromArray(transform.scale ?? [1, 1, 1]);
      const anchorOffset = anchor.clone().multiply(pivotGroup.scale).applyQuaternion(pivotGroup.quaternion);
      pivotGroup.position.fromArray(transform.position).add(anchorOffset);
    };
    setGizmo = (enabled, mode) => {
      transformControls.setMode(mode as TransformControlsMode);
      transformHelper.visible = enabled;
      if (enabled) transformControls.attach(pivotGroup);
      else transformControls.detach();
    };
    const emitTransform = () => {
      const anchorOffset = anchor.clone().multiply(pivotGroup.scale).applyQuaternion(pivotGroup.quaternion);
      const origin = pivotGroup.position.clone().sub(anchorOffset);
      onTransformChanged({
        position: [origin.x, origin.y, origin.z],
        rotation: [
          THREE.MathUtils.radToDeg(pivotGroup.rotation.x),
          THREE.MathUtils.radToDeg(pivotGroup.rotation.y),
          THREE.MathUtils.radToDeg(pivotGroup.rotation.z)
        ],
        scale: [pivotGroup.scale.x, pivotGroup.scale.y, pivotGroup.scale.z]
      });
    };
    const handleDragging = (event: { value: unknown }) => {
      controls.enabled = !Boolean(event.value);
    };
    const handleTransformCommit = () => {
      emitTransform();
      onTransformCommitted();
    };
    transformControls.addEventListener('objectChange', emitTransform);
    transformControls.addEventListener('dragging-changed', handleDragging);
    transformControls.addEventListener('mouseUp', handleTransformCommit);
    if (packedFrame) setPackedGeometry(packedFrame);
    else setGeometry(points);
    setMesh(mesh);
    setMaterial(pointSize, opacity, showColors);
    setRenderMode(renderMode);
    setTransform(cloudTransform, gizmoAnchor);
    setGizmo(editMode, gizmoMode);
    setCameraFrames(cameraFrames, showCameraFrames);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.06 };
    const pointer = new THREE.Vector2();
    const fitFloor = (selectedIndex: number) => {
      const selected = new THREE.Vector3().fromArray(points[selectedIndex].position);
      let candidates = points
        .map((point, index) => ({ index, distance: selected.distanceTo(new THREE.Vector3().fromArray(point.position)) }))
        .filter((candidate) => candidate.distance < 0.5)
        .map((candidate) => candidate.index);
      if (candidates.length < 30) {
        candidates = points
          .map((point, index) => ({ index, distance: selected.distanceTo(new THREE.Vector3().fromArray(point.position)) }))
          .filter((candidate) => candidate.distance < 1.0)
          .map((candidate) => candidate.index);
      }
      if (candidates.length < 12) {
        onFloorMessage('Not enough nearby points to fit a floor plane. Pick a denser area.');
        return;
      }

      let bestNormal: THREE.Vector3 | null = null;
      let bestScore = 0;
      const attempts = Math.min(160, candidates.length * 2);
      for (let attempt = 0; attempt < attempts; attempt += 1) {
        const first = candidates[(attempt * 17) % candidates.length];
        const second = candidates[(attempt * 43 + 7) % candidates.length];
        const third = candidates[(attempt * 71 + 19) % candidates.length];
        if (first === second || first === third || second === third) continue;
        const a = new THREE.Vector3().fromArray(points[first].position);
        const b = new THREE.Vector3().fromArray(points[second].position);
        const c = new THREE.Vector3().fromArray(points[third].position);
        const normal = b.clone().sub(a).cross(c.clone().sub(a));
        if (normal.lengthSq() < 1e-7) continue;
        normal.normalize();
        let score = 0;
        for (const index of candidates) {
          const value = new THREE.Vector3().fromArray(points[index].position);
          if (Math.abs(normal.dot(value.sub(a))) < 0.025) score += 1;
        }
        if (score > bestScore) {
          bestScore = score;
          bestNormal = normal;
        }
      }
      if (!bestNormal || bestScore < 10) {
        onFloorMessage('A stable floor plane could not be found around that point.');
        return;
      }

      // A plane normal has two valid directions. Choose the one for which most
      // of the scan lies above the clicked plane; this makes a clicked floor the
      // bottom even when the cloud initially arrives upside down.
      const orientedNormal = bestNormal.clone();
      const sideStride = Math.max(1, Math.floor(points.length / 20_000));
      let positiveSide = 0;
      let negativeSide = 0;
      for (let index = 0; index < points.length; index += sideStride) {
        const distance = orientedNormal.dot(new THREE.Vector3().fromArray(points[index].position).sub(selected));
        if (distance > 0.04) positiveSide += 1;
        else if (distance < -0.04) negativeSide += 1;
      }
      if (negativeSide > positiveSide) orientedNormal.negate();
      const aboveCount = Math.max(positiveSide, negativeSide);
      const classifiedCount = Math.max(positiveSide + negativeSide, 1);

      pivotGroup.updateMatrixWorld(true);
      const selectedWorld = cloud.localToWorld(selected.clone());
      const normalWorld = orientedNormal
        .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(cloud.matrixWorld))
        .normalize();
      const correction = new THREE.Quaternion().setFromUnitVectors(normalWorld, new THREE.Vector3(0, 1, 0));
      const cloudOriginWorld = cloud.localToWorld(new THREE.Vector3());
      const position = cloudOriginWorld.sub(selectedWorld).applyQuaternion(correction).add(selectedWorld);
      position.y -= selectedWorld.y;
      const rotation = new THREE.Euler().setFromQuaternion(correction.multiply(pivotGroup.quaternion), 'XYZ');
      onFloorDetected({
        position: [position.x, position.y, position.z],
        rotation: [
          THREE.MathUtils.radToDeg(rotation.x),
          THREE.MathUtils.radToDeg(rotation.y),
          THREE.MathUtils.radToDeg(rotation.z)
        ],
        scale: [pivotGroup.scale.x, pivotGroup.scale.y, pivotGroup.scale.z]
      });
      onFloorMessage(`Floor aligned; ${Math.round(aboveCount / classifiedCount * 100)}% of classified points are above it.`);
    };

    const handlePointer = (event: PointerEvent) => {
      if ((!floorPickMode && !anchorPickMode) || points.length === 0) return;
      const bounds = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = activeRenderMode === 'mesh'
        ? raycaster.intersectObject(meshGroup, true)[0]
        : raycaster.intersectObject(cloud, false)[0];
      if (hit) {
        let selectedIndex = hit.index;
        let selectedPoint: [number, number, number] | null = null;
        if (activeRenderMode === 'mesh') {
          const localPoint = meshGroup.worldToLocal(hit.point.clone());
          selectedPoint = [localPoint.x, localPoint.y, localPoint.z];
          let closestDistance = Number.POSITIVE_INFINITY;
          points.forEach((point, index) => {
            const distance = localPoint.distanceToSquared(new THREE.Vector3().fromArray(point.position));
            if (distance < closestDistance) {
              closestDistance = distance;
              selectedIndex = index;
            }
          });
        } else if (selectedIndex !== undefined) {
          selectedPoint = [...points[selectedIndex].position];
        }
        if (selectedIndex !== undefined && selectedPoint) {
          if (anchorPickMode) onAnchorPicked(selectedPoint);
          else fitFloor(selectedIndex);
        }
      } else {
        onFloorMessage(anchorPickMode
          ? 'No surface selected. Click directly on a dense part of the model.'
          : 'No surface selected. Click directly on a dense floor patch.');
      }
    };
    canvas.addEventListener('pointerup', handlePointer);

    let resizeFrame = 0;
    let renderedWidth = 0;
    let renderedHeight = 0;
    const resize = () => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(() => {
        const parent = canvas.parentElement;
        if (!parent) return;
        const bounds = parent.getBoundingClientRect();
        const width = Math.max(1, Math.round(bounds.width));
        const height = Math.max(1, Math.round(bounds.height));
        if (width === renderedWidth && height === renderedHeight) return;
        renderedWidth = width;
        renderedHeight = height;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      });
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas.parentElement!);
    resize();

    let animationFrame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      cancelAnimationFrame(resizeFrame);
      canvas.removeEventListener('pointerup', handlePointer);
      resizeObserver.disconnect();
      controls.dispose();
      transformControls.removeEventListener('objectChange', emitTransform);
      transformControls.removeEventListener('dragging-changed', handleDragging);
      transformControls.removeEventListener('mouseUp', handleTransformCommit);
      transformControls.detach();
      transformControls.dispose();
      scene.remove(transformHelper);
      meshLoadGeneration += 1;
      clearMesh();
      meshMaterial.dispose();
      geometry.dispose();
      material.dispose();
      frustumGeometry.dispose();
      frustumMaterial.dispose();
      renderer.dispose();
    };
  });
</script>

<div class:processing class:point-pick={floorPickMode || anchorPickMode} class="viewer">
  <canvas bind:this={canvas} aria-label="Interactive 3D preview"></canvas>
  <div class:live class="viewer-hud top-left">
    <span class="pulse"></span>
    {processing ? 'Reconstructing phases' : live ? 'Live sensor point cloud' : renderMode === 'mesh' && mesh ? 'Textured mesh view' : points.length ? 'Point-cloud view' : 'Awaiting sensor'}
  </div>
  {#if floorPickMode}
    <div class="viewer-hud floor-hint">Click a dense patch of floor</div>
  {:else if anchorPickMode}
    <div class="viewer-hud floor-hint">Click the model to place the gizmo anchor</div>
  {/if}
  {#if renderMode === 'mesh' ? mesh : packedFrame ? packedFrame.pointCount > 0 : points.length > 0}
    <div class="viewer-hud bottom-right">Drag to orbit · Scroll to zoom{showCameraFrames && cameraFrames.length ? ` · ${cameraFrames.length} camera poses` : ''}</div>
  {:else}
    <div class="empty-state">
      <strong>{renderMode === 'mesh' ? 'No reconstructed mesh yet' : 'No live depth points yet'}</strong>
      <span>{renderMode === 'mesh' ? 'Build the 3D model to create a textured mesh.' : 'The preview starts when the selected sensor streams.'}</span>
    </div>
  {/if}
  {#if processing}
    <div class="processing-scan"></div>
  {/if}
</div>

<style>
  .viewer { position: relative; width: 100%; height: 100%; min-height: 0; overflow: hidden; border-radius: 22px; background: #07111c; box-shadow: inset 0 0 0 1px rgba(139, 193, 216, 0.08); }
  canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .point-pick canvas, .point-pick canvas:active { cursor: crosshair; }
  .viewer-hud { position: absolute; display: flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid rgba(157, 204, 223, 0.12); border-radius: 10px; background: rgba(5, 15, 25, 0.68); color: #adc3d0; font-size: 11px; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase; backdrop-filter: blur(12px); pointer-events: none; }
  .top-left { top: 16px; left: 16px; }
  .bottom-right { right: 16px; bottom: 16px; text-transform: none; letter-spacing: 0; }
  .floor-hint { left: 50%; bottom: 16px; transform: translateX(-50%); border-color: rgba(240, 183, 107, 0.35); color: #f0c68f; }
  .pulse { width: 7px; height: 7px; border-radius: 50%; background: #58d5b5; box-shadow: 0 0 0 4px rgba(88, 213, 181, 0.1); }
  .processing .pulse { background: #f0b76b; animation: pulse 1.1s infinite; }
  .viewer-hud.live .pulse { animation: pulse 1.1s infinite; }
  .empty-state { position: absolute; inset: 0; display: grid; place-content: center; gap: 8px; color: #6c8593; text-align: center; pointer-events: none; }
  .empty-state strong { color: #a9bec8; font-size: 15px; }
  .empty-state span { font-size: 11px; }
  .processing-scan { position: absolute; inset: 0; background: linear-gradient(180deg, transparent 0%, rgba(72, 177, 209, 0.08) 48%, rgba(103, 220, 197, 0.28) 50%, rgba(72, 177, 209, 0.08) 52%, transparent 100%); transform: translateY(-100%); animation: scan 2.4s ease-in-out infinite; pointer-events: none; }
  @keyframes scan { to { transform: translateY(100%); } }
  @keyframes pulse { 50% { opacity: 0.4; transform: scale(0.8); } }
</style>
