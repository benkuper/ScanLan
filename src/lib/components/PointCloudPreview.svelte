<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import type { SparkRenderer as SparkRendererInstance, SplatMesh as SplatMeshInstance } from '@sparkjsdev/spark';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import type { TransformControlsMode } from 'three/examples/jsm/controls/TransformControls.js';
  import { AnchorAngleTransformControls } from '../controls/AnchorAngleTransformControls';
  import type { CloudTransform, MeshViewMode, PackedPreviewFrame, PreviewMesh, PreviewPoint } from '../types';

  type RenderMode = 'points' | 'mesh' | 'splat';
  type AssetLoadingState = 'points' | 'mesh' | 'mesh-texture' | 'splat' | 'splat-gpu' | null;

  export let points: PreviewPoint[] = [];
  export let packedFrame: PackedPreviewFrame | null = null;
  export let processing = false;
  export let live = false;
  export let liveLabel = 'Live RGB-D reconstruction';
  export let emptyDetail = '';
  export let pointSize = 0.034;
  export let opacity = 0.92;
  export let showColors = true;
  export let meshViewMode: MeshViewMode = 'surface';
  export let renderMode: RenderMode = 'points';
  export let mesh: PreviewMesh | null = null;
  export let splatBytes: Uint8Array | null = null;
  export let assetLoading: 'points' | 'mesh' | 'splat' | null = null;
  export let floorPickMode = false;
  export let cloudTransform: CloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
  export let gizmoAnchor: [number, number, number] = [0, 0, 0];
  export let editMode = false;
  export let gizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
  export let rotationSnapDegrees = 0;
  export let onFloorDetected: (transform: CloudTransform) => void = () => undefined;
  export let onFloorMessage: (message: string) => void = () => undefined;
  export let onTransformChanged: (transform: CloudTransform) => void = () => undefined;
  export let onTransformCommitted: () => void = () => undefined;

  let canvas: HTMLCanvasElement;
  let setPoints: (next: PreviewPoint[]) => void = () => undefined;
  let setPackedPoints: (next: PackedPreviewFrame) => void = () => undefined;
  let setMaterial: (size: number, alpha: number, colors: boolean) => void = () => undefined;
  let setMesh: (next: PreviewMesh | null) => void = () => undefined;
  let setMeshMode: (next: MeshViewMode) => void = () => undefined;
  let setSplat: (next: Uint8Array | null) => void = () => undefined;
  let setRenderMode: (next: RenderMode) => void = () => undefined;
  let setTransform: (transform: CloudTransform, anchor: [number, number, number]) => void = () => undefined;
  let setGizmo: (enabled: boolean, mode: 'translate' | 'rotate' | 'scale') => void = () => undefined;
  let setRotationSnap: (degrees: number) => void = () => undefined;
  let splatReady = false;
  let splatError = '';
  let splatLoadProgress: number | null = null;
  let meshTextureReady = true;
  let visibleAssetLoading: AssetLoadingState = null;
  let trackedAssetLoading: AssetLoadingState = null;
  let loadingSince = 0;
  let loadingElapsed = 0;

  $: visibleAssetLoading = assetLoading
    ?? (renderMode === 'mesh' && mesh && !meshTextureReady
      ? 'mesh-texture'
      : renderMode === 'splat' && splatBytes && !splatReady
        ? 'splat-gpu'
        : null);
  $: if (visibleAssetLoading !== trackedAssetLoading) {
    trackedAssetLoading = visibleAssetLoading;
    loadingSince = visibleAssetLoading ? performance.now() : 0;
    loadingElapsed = 0;
  }
  $: {
    if (packedFrame) setPackedPoints(packedFrame);
    else setPoints(points);
  }
  $: setMaterial(pointSize, opacity, showColors);
  $: setMesh(mesh);
  $: setMeshMode(meshViewMode);
  $: setSplat(splatBytes);
  $: setRenderMode(renderMode);
  $: setTransform(cloudTransform, gizmoAnchor);
  $: setGizmo(editMode, gizmoMode);
  $: setRotationSnap(rotationSnapDegrees);

  onMount(() => {
    const loadingTimer = window.setInterval(() => {
      if (loadingSince) loadingElapsed = Math.floor((performance.now() - loadingSince) / 1000);
    }, 250);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07111c');
    scene.fog = new THREE.FogExp2('#07111c', 0.055);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    camera.position.set(6.8, 4.7, 7.6);
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance'
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const qualityPixelRatio = Math.min(window.devicePixelRatio, 1.5);
    const interactionPixelRatio = Math.min(window.devicePixelRatio, 1);
    let appliedPixelRatio = Math.min(window.devicePixelRatio, 2);
    const applyPixelRatio = (next: number) => {
      if (Math.abs(next - appliedPixelRatio) < 0.01) return;
      appliedPixelRatio = next;
      renderer.setPixelRatio(next);
      invalidate();
    };
    renderer.setPixelRatio(appliedPixelRatio);

    let renderInvalidated = true;
    const invalidate = () => {
      renderInvalidated = true;
    };
    let disposed = false;
    let sparkRenderer: SparkRendererInstance | null = null;
    let sparkModulePromise: Promise<typeof import('@sparkjsdev/spark')> | null = null;
    const ensureSpark = async () => {
      sparkModulePromise ??= import('@sparkjsdev/spark');
      const module = await sparkModulePromise;
      if (disposed) return null;
      if (!sparkRenderer) {
        sparkRenderer = new module.SparkRenderer({
          renderer,
          onDirty: invalidate,
          maxStdDev: Math.sqrt(8),
          minSortIntervalMs: 16,
          lodSplatCount: 2_500_000,
          lodRenderScale: 1
        });
        scene.add(sparkRenderer);
      }
      return module;
    };

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.target.set(0, 1.1, 0);
    controls.minDistance = 0.15;
    controls.maxDistance = 30;
    let restoreQualityTimer = 0;
    const interactionStart = () => {
      window.clearTimeout(restoreQualityTimer);
      if (renderMode === 'splat' && sparkRenderer) sparkRenderer.maxStdDev = Math.sqrt(5);
      applyPixelRatio(interactionPixelRatio);
    };
    const interactionEnd = () => {
      window.clearTimeout(restoreQualityTimer);
      restoreQualityTimer = window.setTimeout(() => {
        if (sparkRenderer) sparkRenderer.maxStdDev = Math.sqrt(8);
        applyPixelRatio(renderMode === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
      }, 120);
    };
    controls.addEventListener('change', invalidate);
    controls.addEventListener('start', interactionStart);
    controls.addEventListener('end', interactionEnd);

    const grid = new THREE.GridHelper(12, 24, '#19384a', '#102a39');
    grid.position.y = -0.015;
    scene.add(grid);
    const fillLight = new THREE.HemisphereLight('#d9f3ff', '#17232b', 1.15);
    const keyLight = new THREE.DirectionalLight('#fff1d6', 3.2);
    keyLight.position.set(3, 6, 4);
    scene.add(fillLight, keyLight);

    const root = new THREE.Group();
    scene.add(root);
    const anchor = new THREE.Vector3();

    const transformControls = new AnchorAngleTransformControls(camera, renderer.domElement);
    const transformHelper = transformControls.getHelper();
    transformControls.setSpace('world');
    transformControls.setSize(0.82);
    transformControls.translationSnap = 0.01;
    transformControls.rotationSnap = null;
    transformControls.scaleSnap = 0.01;
    transformHelper.visible = false;
    scene.add(transformHelper);

    const pointGeometry = new THREE.BufferGeometry();
    const pointMaterial = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: showColors,
      color: '#a9dce8',
      sizeAttenuation: true,
      transparent: opacity < 1,
      depthWrite: opacity >= 0.98,
      opacity
    });
    const pointCloud = new THREE.Points(pointGeometry, pointMaterial);
    root.add(pointCloud);

    const meshGroup = new THREE.Group();
    root.add(meshGroup);
    const surfaceMaterial = new THREE.MeshBasicMaterial({
      color: '#a9dce8',
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1
    });
    const shadedMaterial = new THREE.MeshStandardMaterial({
      color: '#93c6d4',
      roughness: 0.78,
      metalness: 0.02,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1
    });
    const wireMaterial = new THREE.MeshBasicMaterial({
      color: '#69cbea',
      side: THREE.DoubleSide,
      wireframe: true,
      transparent: true,
      depthWrite: false,
      opacity
    });
    let meshSurface: THREE.Mesh | null = null;
    let meshWireframe: THREE.Mesh | null = null;
    let meshGeometry: THREE.BufferGeometry | null = null;
    let meshTexture: THREE.Texture | null = null;
    let meshBitmap: ImageBitmap | null = null;
    let meshTextureUrl: string | null = null;
    let meshHasVertexColors = false;
    let meshColorsEnabled = showColors;
    let meshGeneration = 0;
    let appliedMesh: PreviewMesh | null = null;
    let activeMeshMode: MeshViewMode = meshViewMode;

    const updateMeshAppearance = () => {
      const shaded = activeMeshMode === 'shaded';
      const showSurface = activeMeshMode !== 'wireframe';
      const requiresTexture = !shaded && meshColorsEnabled && !meshHasVertexColors && Boolean(meshTexture);
      if (meshSurface) {
        meshSurface.material = shaded ? shadedMaterial : surfaceMaterial;
        meshSurface.visible = showSurface && (!requiresTexture || meshTextureReady);
      }
      if (meshWireframe) {
        meshWireframe.visible = activeMeshMode === 'wireframe' || activeMeshMode === 'surface-wireframe';
      }
      wireMaterial.color.set(activeMeshMode === 'wireframe' ? '#79daf7' : '#153e50');
      wireMaterial.opacity = activeMeshMode === 'wireframe' ? opacity : Math.min(0.82, Math.max(0.35, opacity));
      fillLight.visible = renderMode === 'mesh' && shaded;
      keyLight.visible = fillLight.visible;
      invalidate();
    };

    const clearMesh = () => {
      meshGroup.clear();
      meshGeometry?.dispose();
      meshGeometry = null;
      meshSurface = null;
      meshWireframe = null;
      meshTexture?.dispose();
      meshTexture = null;
      meshBitmap?.close();
      meshBitmap = null;
      if (meshTextureUrl) URL.revokeObjectURL(meshTextureUrl);
      meshTextureUrl = null;
      meshHasVertexColors = false;
      surfaceMaterial.map = null;
      surfaceMaterial.vertexColors = false;
      shadedMaterial.vertexColors = false;
      surfaceMaterial.needsUpdate = true;
      shadedMaterial.needsUpdate = true;
      meshTextureReady = true;
      invalidate();
    };

    const applyTexture = (texture: THREE.Texture, generation: number, bitmap: ImageBitmap | null = null) => {
      if (generation !== meshGeneration) {
        texture.dispose();
        bitmap?.close();
        return;
      }
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy());
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      if (bitmap) texture.flipY = false;
      texture.needsUpdate = true;
      meshTexture = texture;
      meshBitmap = bitmap;
      meshTextureReady = true;
      surfaceMaterial.map = meshColorsEnabled ? texture : null;
      surfaceMaterial.needsUpdate = true;
      updateMeshAppearance();
    };

    setMesh = (next) => {
      if (next === appliedMesh) return;
      appliedMesh = next;
      meshGeneration += 1;
      const generation = meshGeneration;
      clearMesh();
      if (!next) return;

      meshGeometry = new THREE.BufferGeometry();
      meshGeometry.setAttribute('position', new THREE.BufferAttribute(next.positions, 3));
      if (next.uvs) meshGeometry.setAttribute('uv', new THREE.BufferAttribute(next.uvs, 2));
      if (next.colors) {
        meshGeometry.setAttribute('color', new THREE.BufferAttribute(next.colors, 3, true));
        meshHasVertexColors = true;
      }
      meshGeometry.setIndex(new THREE.BufferAttribute(next.indices, 1));
      meshGeometry.computeVertexNormals();
      meshGeometry.computeBoundingSphere();
      meshSurface = new THREE.Mesh(meshGeometry, surfaceMaterial);
      meshWireframe = new THREE.Mesh(meshGeometry, wireMaterial);
      meshWireframe.renderOrder = 2;
      meshGroup.add(meshSurface, meshWireframe);

      surfaceMaterial.vertexColors = meshHasVertexColors && meshColorsEnabled;
      shadedMaterial.vertexColors = meshHasVertexColors && meshColorsEnabled;
      surfaceMaterial.color.set(surfaceMaterial.vertexColors ? '#ffffff' : '#a9dce8');
      shadedMaterial.color.set(shadedMaterial.vertexColors ? '#ffffff' : '#93c6d4');
      surfaceMaterial.needsUpdate = true;
      shadedMaterial.needsUpdate = true;
      updateMeshAppearance();

      if (!next.texture?.byteLength) return;
      meshTextureReady = false;
      const bytes = next.texture.byteOffset === 0 && next.texture.byteLength === next.texture.buffer.byteLength
        ? next.texture.buffer as ArrayBuffer
        : next.texture.slice().buffer as ArrayBuffer;
      const blob = new Blob([bytes], { type: 'image/png' });
      const fallback = () => {
        meshTextureUrl = URL.createObjectURL(blob);
        const url = meshTextureUrl;
        new THREE.TextureLoader().load(
          url,
          (texture) => {
            URL.revokeObjectURL(url);
            if (meshTextureUrl === url) meshTextureUrl = null;
            applyTexture(texture, generation);
          },
          undefined,
          () => {
            URL.revokeObjectURL(url);
            if (meshTextureUrl === url) meshTextureUrl = null;
            if (generation === meshGeneration) {
              meshTextureReady = true;
              updateMeshAppearance();
            }
          }
        );
      };
      if ('createImageBitmap' in window) {
        void createImageBitmap(blob, {
          imageOrientation: 'flipY',
          premultiplyAlpha: 'none',
          colorSpaceConversion: 'none'
        })
          .then((bitmap) => applyTexture(new THREE.Texture(bitmap), generation, bitmap))
          .catch(fallback);
      } else {
        fallback();
      }
    };

    setMeshMode = (next) => {
      activeMeshMode = next;
      updateMeshAppearance();
    };

    const splatGroup = new THREE.Group();
    root.add(splatGroup);
    let loadedSplat: SplatMeshInstance | null = null;
    let appliedSplat: Uint8Array | null = null;
    let splatGeneration = 0;
    const clearSplat = () => {
      splatGeneration += 1;
      if (loadedSplat) {
        splatGroup.remove(loadedSplat);
        loadedSplat.dispose();
        loadedSplat = null;
      }
      splatReady = false;
      splatError = '';
      splatLoadProgress = null;
      invalidate();
    };
    setSplat = (next) => {
      if (next === appliedSplat) return;
      appliedSplat = next;
      if (!next) {
        clearSplat();
        return;
      }
      splatGeneration += 1;
      const generation = splatGeneration;
      splatReady = false;
      splatError = '';
      splatLoadProgress = 0;
      void ensureSpark().then((module) => {
        if (!module || generation !== splatGeneration) return;
        const previous = loadedSplat;
        const candidate = new module.SplatMesh({
          fileBytes: next,
          fileName: 'scanlan-preview.splat',
          editable: false,
          raycastable: false,
          lod: 'quality',
          lodAbove: 2_500_000,
          nonLod: false,
          onProgress: (event) => {
            if (generation === splatGeneration && event.lengthComputable && event.total > 0) {
              splatLoadProgress = THREE.MathUtils.clamp(event.loaded / event.total, 0, 1);
            }
          }
        });
        candidate.visible = false;
        candidate.opacity = opacity;
        candidate.recolor.set(showColors ? '#ffffff' : '#a9dce8');
        splatGroup.add(candidate);
        invalidate();
        void candidate.initialized.then(() => {
          if (generation !== splatGeneration) {
            splatGroup.remove(candidate);
            candidate.dispose();
            return;
          }
          if (previous) {
            splatGroup.remove(previous);
            previous.dispose();
          }
          loadedSplat = candidate;
          candidate.visible = true;
          splatReady = true;
          splatLoadProgress = 1;
          invalidate();
        }).catch((error: unknown) => {
          splatGroup.remove(candidate);
          candidate.dispose();
          if (generation !== splatGeneration) return;
          splatReady = Boolean(loadedSplat);
          splatLoadProgress = null;
          splatError = error instanceof Error ? error.message : String(error);
          invalidate();
        });
      }).catch((error: unknown) => {
        if (generation !== splatGeneration) return;
        splatReady = Boolean(loadedSplat);
        splatLoadProgress = null;
        splatError = error instanceof Error ? error.message : String(error);
        invalidate();
      });
    };

    setPoints = (next) => {
      const positions = new Float32Array(next.length * 3);
      const colors = new Uint8Array(next.length * 3);
      next.forEach((point, index) => {
        positions.set(point.position, index * 3);
        colors.set(point.color, index * 3);
      });
      pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      pointGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
      pointGeometry.computeBoundingSphere();
      invalidate();
    };
    setPackedPoints = (next) => {
      pointGeometry.setAttribute('position', new THREE.BufferAttribute(next.positions, 3));
      pointGeometry.setAttribute('color', new THREE.BufferAttribute(next.colors, 3, true));
      pointGeometry.computeBoundingSphere();
      invalidate();
    };
    setMaterial = (size, alpha, colors) => {
      pointMaterial.size = size;
      pointMaterial.opacity = alpha;
      pointMaterial.transparent = alpha < 1;
      pointMaterial.depthWrite = alpha >= 0.98;
      if (pointMaterial.vertexColors !== colors) {
        pointMaterial.vertexColors = colors;
        pointMaterial.needsUpdate = true;
      }
      meshColorsEnabled = colors;
      for (const material of [surfaceMaterial, shadedMaterial]) {
        material.opacity = alpha;
        material.transparent = alpha < 1;
      }
      surfaceMaterial.map = colors ? meshTexture : null;
      surfaceMaterial.vertexColors = colors && meshHasVertexColors;
      shadedMaterial.vertexColors = colors && meshHasVertexColors;
      surfaceMaterial.needsUpdate = true;
      shadedMaterial.needsUpdate = true;
      wireMaterial.opacity = alpha;
      if (loadedSplat) {
        loadedSplat.opacity = alpha;
        loadedSplat.recolor.set(colors ? '#ffffff' : '#a9dce8');
      }
      updateMeshAppearance();
      invalidate();
    };
    let activeRenderMode: RenderMode = renderMode;
    setRenderMode = (next) => {
      activeRenderMode = next;
      pointCloud.visible = next === 'points';
      meshGroup.visible = next === 'mesh';
      splatGroup.visible = next === 'splat';
      grid.visible = next !== 'splat';
      applyPixelRatio(next === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
      updateMeshAppearance();
      invalidate();
    };

    setTransform = (transform, nextAnchor) => {
      anchor.fromArray(nextAnchor);
      const inverseAnchor = anchor.clone().multiplyScalar(-1);
      pointCloud.position.copy(inverseAnchor);
      meshGroup.position.copy(inverseAnchor);
      splatGroup.position.copy(inverseAnchor);
      root.rotation.set(
        THREE.MathUtils.degToRad(transform.rotation[0]),
        THREE.MathUtils.degToRad(transform.rotation[1]),
        THREE.MathUtils.degToRad(transform.rotation[2]),
        'XYZ'
      );
      root.scale.fromArray(transform.scale);
      const anchorOffset = anchor.clone().multiply(root.scale).applyQuaternion(root.quaternion);
      root.position.fromArray(transform.position).add(anchorOffset);
      invalidate();
    };
    let activeGizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
    let gizmoAttached = false;
    setGizmo = (enabled, mode) => {
      if (mode !== activeGizmoMode) {
        activeGizmoMode = mode;
        transformControls.setMode(mode as TransformControlsMode);
      }
      if (enabled !== gizmoAttached) {
        gizmoAttached = enabled;
        transformHelper.visible = enabled;
        if (enabled) {
          transformControls.attach(root);
          // Keep a stable interaction resolution for the whole edit session.
          // Resizing the WebGL target after every mouse-up caused the release stall.
          applyPixelRatio(interactionPixelRatio);
        } else {
          transformControls.detach();
          interactionEnd();
        }
      }
      invalidate();
    };
    setRotationSnap = (degrees) => {
      transformControls.rotationSnap = degrees > 0 ? THREE.MathUtils.degToRad(degrees) : null;
    };
    const emitTransform = () => {
      const anchorOffset = anchor.clone().multiply(root.scale).applyQuaternion(root.quaternion);
      const origin = root.position.clone().sub(anchorOffset);
      onTransformChanged({
        position: [origin.x, origin.y, origin.z],
        rotation: [
          THREE.MathUtils.radToDeg(root.rotation.x),
          THREE.MathUtils.radToDeg(root.rotation.y),
          THREE.MathUtils.radToDeg(root.rotation.z)
        ],
        scale: [root.scale.x, root.scale.y, root.scale.z]
      });
      invalidate();
    };
    const handleGizmoDragging = (event: { value: unknown }) => {
      const dragging = Boolean(event.value);
      controls.enabled = !dragging;
      if (dragging) interactionStart();
      else if (!editMode) interactionEnd();
    };
    const commitTransform = () => {
      emitTransform();
      onTransformCommitted();
    };
    // TransformControls already edits `root` directly. Sending every pointer
    // event through Svelte only writes the same matrix back and makes large
    // scenes feel CPU-bound. Publish the final pose once on mouse-up instead.
    transformControls.addEventListener('objectChange', invalidate);
    transformControls.addEventListener('dragging-changed', handleGizmoDragging);
    transformControls.addEventListener('mouseUp', commitTransform);

    const sourcePointCount = () => points.length || Math.floor((mesh?.positions.length ?? 0) / 3);
    const sourcePoint = (index: number, target = new THREE.Vector3()) => {
      if (points.length) return target.fromArray(points[index].position);
      const positions = mesh?.positions;
      return positions
        ? target.set(positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2])
        : target.set(0, 0, 0);
    };
    const fitFloor = (selected: THREE.Vector3, contentObject: THREE.Object3D) => {
      const count = sourcePointCount();
      const stride = Math.max(1, Math.floor(count / 120_000));
      const collectNearby = (radius: number) => {
        const radiusSquared = radius * radius;
        const nearby: THREE.Vector3[] = [];
        for (let index = 0; index < count; index += stride) {
          const candidate = sourcePoint(index);
          if (candidate.distanceToSquared(selected) < radiusSquared) nearby.push(candidate);
        }
        if (nearby.length <= 8_000) return nearby;
        const compactStride = Math.ceil(nearby.length / 8_000);
        return nearby.filter((_, index) => index % compactStride === 0);
      };
      let candidates = collectNearby(0.5);
      if (candidates.length < 30) candidates = collectNearby(1.0);
      if (candidates.length < 12) {
        onFloorMessage('Not enough nearby geometry to fit a floor plane. Pick a denser area.');
        return;
      }

      let bestNormal: THREE.Vector3 | null = null;
      let bestScore = 0;
      const attempts = Math.min(160, candidates.length * 2);
      for (let attempt = 0; attempt < attempts; attempt += 1) {
        const a = candidates[(attempt * 17) % candidates.length];
        const b = candidates[(attempt * 43 + 7) % candidates.length];
        const c = candidates[(attempt * 71 + 19) % candidates.length];
        if (a === b || a === c || b === c) continue;
        const normal = b.clone().sub(a).cross(c.clone().sub(a));
        if (normal.lengthSq() < 1e-7) continue;
        normal.normalize();
        let score = 0;
        for (const candidate of candidates) {
          if (Math.abs(normal.dot(candidate.clone().sub(a))) < 0.025) score += 1;
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

      const orientedNormal = bestNormal.clone();
      const sideStride = Math.max(1, Math.floor(count / 20_000));
      let positiveSide = 0;
      let negativeSide = 0;
      for (let index = 0; index < count; index += sideStride) {
        const distance = orientedNormal.dot(sourcePoint(index).sub(selected));
        if (distance > 0.04) positiveSide += 1;
        else if (distance < -0.04) negativeSide += 1;
      }
      if (negativeSide > positiveSide) orientedNormal.negate();
      const aboveCount = Math.max(positiveSide, negativeSide);
      const classifiedCount = Math.max(positiveSide + negativeSide, 1);

      root.updateMatrixWorld(true);
      const selectedWorld = contentObject.localToWorld(selected.clone());
      const normalWorld = orientedNormal
        .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(contentObject.matrixWorld))
        .normalize();
      const correction = new THREE.Quaternion().setFromUnitVectors(
        normalWorld,
        new THREE.Vector3(0, 1, 0)
      );
      const modelOriginWorld = contentObject.localToWorld(new THREE.Vector3());
      const position = modelOriginWorld
        .sub(selectedWorld)
        .applyQuaternion(correction)
        .add(selectedWorld);
      position.y -= selectedWorld.y;
      const rotation = new THREE.Euler().setFromQuaternion(
        correction.multiply(root.quaternion),
        'XYZ'
      );
      onFloorDetected({
        position: [position.x, position.y, position.z],
        rotation: [
          THREE.MathUtils.radToDeg(rotation.x),
          THREE.MathUtils.radToDeg(rotation.y),
          THREE.MathUtils.radToDeg(rotation.z)
        ],
        scale: [root.scale.x, root.scale.y, root.scale.z]
      });
      onFloorMessage(
        `Floor aligned; ${Math.round(aboveCount / classifiedCount * 100)}% of classified geometry is above it.`
      );
    };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.06 };
    const pointer = new THREE.Vector2();
    const handleFloorPick = (event: PointerEvent) => {
      if (!floorPickMode || activeRenderMode === 'splat' || sourcePointCount() === 0) return;
      const bounds = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const contentObject = activeRenderMode === 'mesh' ? meshGroup : pointCloud;
      const hit = raycaster.intersectObject(contentObject, true)[0];
      if (!hit) {
        onFloorMessage('No surface selected. Click directly on a dense floor patch.');
        return;
      }
      fitFloor(contentObject.worldToLocal(hit.point.clone()), contentObject);
    };
    canvas.addEventListener('pointerup', handleFloorPick);

    if (packedFrame) setPackedPoints(packedFrame);
    else setPoints(points);
    setMesh(mesh);
    setSplat(splatBytes);
    setMaterial(pointSize, opacity, showColors);
    setMeshMode(meshViewMode);
    setRenderMode(renderMode);
    setTransform(cloudTransform, gizmoAnchor);
    setGizmo(editMode, gizmoMode);
    setRotationSnap(rotationSnapDegrees);

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
        invalidate();
      });
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas.parentElement!);
    resize();

    let animationFrame = 0;
    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      const changed = controls.update();
      if (!changed && !renderInvalidated) return;
      renderInvalidated = false;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      cancelAnimationFrame(resizeFrame);
      window.clearInterval(loadingTimer);
      window.clearTimeout(restoreQualityTimer);
      canvas.removeEventListener('pointerup', handleFloorPick);
      resizeObserver.disconnect();
      controls.removeEventListener('change', invalidate);
      controls.removeEventListener('start', interactionStart);
      controls.removeEventListener('end', interactionEnd);
      controls.dispose();
      transformControls.removeEventListener('objectChange', invalidate);
      transformControls.removeEventListener('dragging-changed', handleGizmoDragging);
      transformControls.removeEventListener('mouseUp', commitTransform);
      transformControls.detach();
      transformControls.dispose();
      scene.remove(transformHelper);
      clearSplat();
      clearMesh();
      sparkRenderer?.dispose();
      pointGeometry.dispose();
      pointMaterial.dispose();
      surfaceMaterial.dispose();
      shadedMaterial.dispose();
      wireMaterial.dispose();
      renderer.dispose();
    };
  });
</script>

<div class:processing class:point-pick={floorPickMode} class="viewer">
  <canvas bind:this={canvas} aria-label="Interactive 3D reconstruction"></canvas>
  <div class:live class="viewer-hud top-left">
    <span class="pulse"></span>
    {processing ? renderMode === 'splat' ? 'Training 2D Gaussian splats' : 'Reconstructing geometry' : live ? liveLabel : renderMode === 'splat' ? splatError ? 'Splat preview failed' : splatReady ? '2D Gaussian splat' : 'Loading splat' : renderMode === 'mesh' && mesh ? meshViewMode === 'wireframe' ? 'Wireframe' : meshViewMode === 'shaded' ? 'Shaded mesh' : meshViewMode === 'surface-wireframe' ? 'Mesh + wireframe' : 'Textured mesh' : points.length || packedFrame?.pointCount ? 'Point cloud' : 'Awaiting RGB-D frames'}
  </div>

  {#if floorPickMode}
    <div class="viewer-hud floor-hint">Click a dense patch of floor</div>
  {:else if editMode}
    <div class="viewer-hud floor-hint">Drag the gizmo · W move · E rotate · R scale</div>
  {/if}

  {#if visibleAssetLoading}
    <div class="asset-loading" aria-live="polite" aria-busy="true">
      <div class="asset-loading-spinner"></div>
      <strong>{visibleAssetLoading === 'points' ? 'Loading point cloud' : visibleAssetLoading === 'mesh' ? 'Loading mesh' : visibleAssetLoading === 'mesh-texture' ? 'Uploading texture' : visibleAssetLoading === 'splat' ? 'Loading Gaussian data' : 'Preparing Gaussian renderer'}</strong>
      <span>{visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null ? `${Math.round(splatLoadProgress * 100)}% · ` : ''}{loadingElapsed}s</span>
      <div class:determinate={visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null} class="asset-loading-track"><i style={visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null ? `width: ${Math.max(2, splatLoadProgress * 100)}%` : undefined}></i></div>
    </div>
  {/if}

  {#if renderMode === 'splat' ? splatReady : renderMode === 'mesh' ? mesh : packedFrame ? packedFrame.pointCount > 0 : points.length > 0}
    <div class="viewer-hud bottom-right">Drag to orbit · Scroll to zoom</div>
  {:else}
    <div class="empty-state">
      <strong>{processing ? 'Preparing reconstruction geometry…' : renderMode === 'splat' ? splatError || 'No Gaussian splat yet' : renderMode === 'mesh' ? 'No reconstructed mesh yet' : live ? 'No valid depth in camera range' : 'No live depth points yet'}</strong>
      <span>{processing ? 'The viewer updates whenever the worker publishes a quality-gated snapshot.' : live ? emptyDetail || 'Aim the camera at a surface between its minimum range and the configured depth limit.' : 'Start capture or build the selected output.'}</span>
    </div>
  {/if}
  {#if processing}<div class="processing-scan"></div>{/if}
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
  .asset-loading { position: absolute; z-index: 5; left: 50%; top: 50%; width: min(320px, calc(100% - 40px)); padding: 22px 24px; display: grid; justify-items: center; gap: 9px; transform: translate(-50%, -50%); border: 1px solid rgba(112, 186, 215, 0.28); border-radius: 16px; background: rgba(5, 14, 23, 0.94); box-shadow: 0 22px 70px rgba(0, 0, 0, 0.42); color: #b7ccd7; text-align: center; pointer-events: none; }
  .asset-loading strong { color: #d4e7ef; font-size: 14px; }
  .asset-loading span { color: #78909d; font-size: 10px; }
  .asset-loading-spinner { width: 25px; height: 25px; border: 2px solid rgba(104, 195, 227, 0.2); border-top-color: #68c3e3; border-radius: 50%; animation: loading-spin 0.8s linear infinite; }
  .asset-loading-track { width: 100%; height: 3px; margin-top: 4px; overflow: hidden; border-radius: 3px; background: rgba(104, 195, 227, 0.12); }
  .asset-loading-track i { display: block; width: 38%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, transparent, #68c3e3, transparent); animation: loading-track 1.15s ease-in-out infinite; }
  .asset-loading-track.determinate i { background: #68c3e3; animation: none; transition: width 120ms linear; }
  .processing-scan { position: absolute; inset: 0; background: linear-gradient(180deg, transparent 0%, rgba(72, 177, 209, 0.08) 48%, rgba(103, 220, 197, 0.28) 50%, rgba(72, 177, 209, 0.08) 52%, transparent 100%); transform: translateY(-100%); animation: scan 2.4s ease-in-out infinite; pointer-events: none; }
  @keyframes scan { to { transform: translateY(100%); } }
  @keyframes pulse { 50% { opacity: 0.4; transform: scale(0.8); } }
  @keyframes loading-spin { to { transform: rotate(360deg); } }
  @keyframes loading-track { from { transform: translateX(-110%); } to { transform: translateX(270%); } }
</style>
