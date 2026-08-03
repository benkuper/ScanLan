<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { TransformControls, type TransformControlsMode } from 'three/examples/jsm/controls/TransformControls.js';
  import type { CameraFrame, CloudTransform, MeshViewMode, PackedPreviewFrame, PreviewMesh, PreviewPoint } from '../types';

  type AssetLoadingState = 'points' | 'mesh' | 'mesh-texture' | 'splat' | 'splat-gpu' | null;

  export let points: PreviewPoint[] = [];
  export let packedFrame: PackedPreviewFrame | null = null;
  export let processing = false;
  export let live = false;
  export let pointSize = 0.034;
  export let opacity = 0.92;
  export let showColors = true;
  export let meshViewMode: MeshViewMode = 'surface';
  export let lightDirection: [number, number, number] = [0.45, 0.8, 0.35];
  export let lightEditMode = false;
  export let renderMode: 'points' | 'mesh' | 'splat' = 'points';
  export let mesh: PreviewMesh | null = null;
  export let splatBytes: Uint8Array | null = null;
  export let assetLoading: 'points' | 'mesh' | 'splat' | null = null;
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
  export let onLightDirectionChanged: (direction: [number, number, number]) => void = () => undefined;

  let canvas: HTMLCanvasElement;
  let setGeometry: (nextPoints: PreviewPoint[]) => void = () => undefined;
  let setPackedGeometry: (frame: PackedPreviewFrame) => void = () => undefined;
  let setMaterial: (size: number, alpha: number, colors: boolean) => void = () => undefined;
  let setMeshView: (mode: MeshViewMode) => void = () => undefined;
  let setLight: (direction: [number, number, number], editing: boolean) => void = () => undefined;
  let setMesh: (nextMesh: PreviewMesh | null) => void = () => undefined;
  let setSplat: (bytes: Uint8Array | null) => void = () => undefined;
  let setRenderMode: (mode: 'points' | 'mesh' | 'splat') => void = () => undefined;
  let setTransform: (transform: CloudTransform, anchor: [number, number, number]) => void = () => undefined;
  let setGizmo: (enabled: boolean, mode: 'translate' | 'rotate' | 'scale') => void = () => undefined;
  let setCameraFrames: (frames: CameraFrame[], visible: boolean) => void = () => undefined;
  let splatReady = false;
  let splatError = '';
  let splatLoadProgress: number | null = null;
  let meshTextureReady = true;
  let visibleAssetLoading: AssetLoadingState = null;
  let trackedAssetLoading: AssetLoadingState = null;
  let assetLoadingStartedAt = 0;
  let assetLoadingElapsed = 0;

  $: visibleAssetLoading = assetLoading
    ?? (renderMode === 'mesh' && mesh && !meshTextureReady && (meshViewMode === 'surface' || meshViewMode === 'surface-wireframe')
      ? 'mesh-texture'
      : renderMode === 'splat' && splatBytes && !splatReady
        ? 'splat-gpu'
        : null);
  $: if (visibleAssetLoading !== trackedAssetLoading) {
    trackedAssetLoading = visibleAssetLoading;
    assetLoadingStartedAt = performance.now();
    assetLoadingElapsed = 0;
  }

  $: {
    if (packedFrame) setPackedGeometry(packedFrame);
    else setGeometry(points);
  }
  $: setMaterial(pointSize, opacity, showColors);
  $: setMeshView(meshViewMode);
  $: setLight(lightDirection, lightEditMode);
  $: setMesh(mesh);
  $: setSplat(splatBytes);
  $: setRenderMode(renderMode);
  $: setTransform(cloudTransform, gizmoAnchor);
  $: setGizmo(editMode, gizmoMode);
  $: setCameraFrames(cameraFrames, showCameraFrames);

  onMount(() => {
    const loadingElapsedTimer = window.setInterval(() => {
      if (visibleAssetLoading) {
        assetLoadingElapsed = Math.max(0, Math.floor((performance.now() - assetLoadingStartedAt) / 1000));
      }
    }, 250);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07111c');
    scene.fog = new THREE.FogExp2('#07111c', 0.055);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    camera.position.set(6.8, 4.7, 7.6);

    let activeRenderMode: 'points' | 'mesh' | 'splat' = renderMode;
    let activeMeshViewMode: MeshViewMode = meshViewMode;
    let renderRequested = true;
    const requestRender = () => {
      renderRequested = true;
    };
    const qualityPixelRatio = Math.min(window.devicePixelRatio, 1.5);
    const interactionPixelRatio = Math.min(window.devicePixelRatio, 1);
    let appliedPixelRatio = activeRenderMode === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2);
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(appliedPixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const sparkRenderer = new SparkRenderer({
      renderer,
      onDirty: requestRender,
      maxStdDev: Math.sqrt(8),
      minSortIntervalMs: 16,
      lodSplatCount: 2_500_000,
      lodRenderScale: 1
    });
    scene.add(sparkRenderer);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.target.set(0, 1.1, 0);
    controls.minDistance = 0.5;
    controls.maxDistance = 30;
    let restoreQualityTimer = 0;
    const setPixelRatio = (next: number) => {
      if (Math.abs(next - appliedPixelRatio) < 0.01) return;
      appliedPixelRatio = next;
      renderer.setPixelRatio(next);
      requestRender();
    };
    const handleInteractionStart = () => {
      window.clearTimeout(restoreQualityTimer);
      if (activeRenderMode === 'splat') {
        sparkRenderer.maxStdDev = Math.sqrt(5);
      }
      setPixelRatio(interactionPixelRatio);
    };
    const handleInteractionEnd = () => {
      window.clearTimeout(restoreQualityTimer);
      restoreQualityTimer = window.setTimeout(() => {
        if (activeRenderMode === 'splat') {
          sparkRenderer.maxStdDev = Math.sqrt(8);
        }
        setPixelRatio(activeRenderMode === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
        requestRender();
      }, 120);
    };
    // OrbitControls applies wheel dolly immediately inside its wheel handler.
    // By the next animation frame controls.update() can already report false,
    // so an on-demand renderer must also invalidate on the change event.
    let updateLightVisual = () => undefined;
    const handleControlsChange = () => {
      updateLightVisual();
      requestRender();
    };
    controls.addEventListener('change', handleControlsChange);
    controls.addEventListener('start', handleInteractionStart);
    controls.addEventListener('end', handleInteractionEnd);

    let smoothZoomActive = false;
    let targetZoomDistance = controls.getDistance();
    let lastZoomInputTime = 0;
    const handleSmoothWheel = (event: WheelEvent) => {
      if (!controls.enabled || !controls.enableZoom) return;
      event.preventDefault();
      event.stopImmediatePropagation();

      const deltaModeScale = event.deltaMode === WheelEvent.DOM_DELTA_LINE
        ? 16
        : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
          ? 100
          : 1;
      const pinchScale = event.ctrlKey ? 10 : 1;
      const delta = event.deltaY * deltaModeScale * pinchScale;
      if (!Number.isFinite(delta) || Math.abs(delta) < 0.001) return;

      if (!smoothZoomActive) {
        targetZoomDistance = controls.getDistance();
        handleInteractionStart();
      }
      // Every momentum tick adjusts the same destination while one continuous
      // camera motion stays active. This responds on the first event without
      // restarting a short animation for every mouse-wheel packet.
      smoothZoomActive = true;
      lastZoomInputTime = performance.now();
      const notchScale = Math.pow(0.95, controls.zoomSpeed * Math.abs(delta * 0.01));
      targetZoomDistance = THREE.MathUtils.clamp(
        delta < 0 ? targetZoomDistance * notchScale : targetZoomDistance / notchScale,
        controls.minDistance,
        controls.maxDistance
      );
      requestRender();
    };
    canvas.addEventListener('wheel', handleSmoothWheel, { passive: false, capture: true });

    const grid = new THREE.GridHelper(12, 24, '#19384a', '#102a39');
    grid.position.y = -0.015;
    scene.add(grid);

    const fillLight = new THREE.HemisphereLight('#d9f3ff', '#17232b', 1.15);
    const keyLight = new THREE.DirectionalLight('#fff1d6', 3.2);
    const lightTarget = new THREE.Object3D();
    keyLight.target = lightTarget;
    scene.add(fillLight, keyLight, lightTarget);

    const lightHandle = new THREE.Object3D();
    const lightHandleGeometry = new THREE.SphereGeometry(0.08, 14, 10);
    const lightHandleMaterial = new THREE.MeshBasicMaterial({ color: '#ffd27c', depthTest: false });
    const lightHandleMarker = new THREE.Mesh(lightHandleGeometry, lightHandleMaterial);
    lightHandleMarker.renderOrder = 10;
    lightHandle.add(lightHandleMarker);
    const lightDirectionGeometry = new THREE.BufferGeometry();
    lightDirectionGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
    const lightDirectionMaterial = new THREE.LineBasicMaterial({ color: '#ffd27c', transparent: true, opacity: 0.82, depthTest: false });
    const lightDirectionLine = new THREE.Line(lightDirectionGeometry, lightDirectionMaterial);
    lightDirectionLine.renderOrder = 9;
    lightHandle.visible = false;
    lightDirectionLine.visible = false;
    scene.add(lightHandle, lightDirectionLine);

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
      // The fused surface is opaque at full opacity and participates normally
      // in depth testing; transparent previews still keep the nearest surface.
      depthWrite: true,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      opacity
    });
    const shadedMeshMaterial = new THREE.MeshStandardMaterial({
      color: '#93c6d4',
      roughness: 0.78,
      metalness: 0.02,
      side: THREE.DoubleSide,
      transparent: opacity < 1,
      depthWrite: true,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      opacity
    });
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: '#69cbea',
      side: THREE.DoubleSide,
      wireframe: true,
      transparent: true,
      depthWrite: false,
      opacity
    });
    let meshSurface: THREE.Mesh | null = null;
    let meshWireframe: THREE.Mesh | null = null;
    pivotGroup.add(meshGroup);
    const splatGroup = new THREE.Group();
    pivotGroup.add(splatGroup);
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

    const lightTransformControls = new TransformControls(camera, renderer.domElement);
    const lightTransformHelper = lightTransformControls.getHelper();
    lightTransformControls.setMode('translate');
    lightTransformControls.setSpace('world');
    lightTransformControls.setSize(0.7);
    lightTransformHelper.visible = false;
    scene.add(lightTransformHelper);

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
      requestRender();
    };
    setPackedGeometry = (frame) => {
      geometry.setAttribute('position', new THREE.BufferAttribute(frame.positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(frame.colors, 3, true));
      geometry.computeBoundingSphere();
      requestRender();
    };
    let loadedMeshTexture: THREE.Texture | null = null;
    let loadedMeshBitmap: ImageBitmap | null = null;
    let pendingTextureUrl: string | null = null;
    let meshLoadGeneration = 0;
    let meshColorsEnabled = showColors;
    let loadedMeshHasVertexColors = false;
    let loadedSplat: SplatMesh | null = null;
    let appliedSplatBytes: Uint8Array | null = null;
    let splatLoadGeneration = 0;
    let lightEditingEnabled = lightEditMode;
    let lightHandleRadius = 2.5;
    let appliedLightDirection = new THREE.Vector3(0.45, 0.8, 0.35).normalize();
    updateLightVisual = () => {
      const center = controls.target;
      lightTarget.position.copy(center);
      keyLight.position.copy(center).addScaledVector(appliedLightDirection, lightHandleRadius * 3);
      lightHandle.position.copy(center).addScaledVector(appliedLightDirection, lightHandleRadius);
      const positions = lightDirectionGeometry.getAttribute('position') as THREE.BufferAttribute;
      positions.setXYZ(0, center.x, center.y, center.z);
      positions.setXYZ(1, lightHandle.position.x, lightHandle.position.y, lightHandle.position.z);
      positions.needsUpdate = true;
      keyLight.target.updateMatrixWorld();
    };
    const refreshLightEditing = () => {
      const visible = lightEditingEnabled && activeRenderMode === 'mesh' && activeMeshViewMode === 'shaded' && Boolean(meshSurface);
      lightHandle.visible = visible;
      lightDirectionLine.visible = visible;
      lightTransformHelper.visible = visible;
      if (visible) lightTransformControls.attach(lightHandle);
      else lightTransformControls.detach();
    };
    const updateMeshAppearance = () => {
      const shaded = activeMeshViewMode === 'shaded';
      const showSurface = activeMeshViewMode !== 'wireframe';
      const textureRequired = !shaded && meshColorsEnabled && !loadedMeshHasVertexColors;
      if (meshSurface) {
        meshSurface.material = shaded ? shadedMeshMaterial : meshMaterial;
        meshSurface.visible = showSurface && (!textureRequired || meshTextureReady);
      }
      if (meshWireframe) {
        meshWireframe.visible = activeMeshViewMode === 'surface-wireframe' || activeMeshViewMode === 'wireframe';
      }
      wireframeMaterial.color.set(activeMeshViewMode === 'wireframe' ? '#79daf7' : '#153e50');
      wireframeMaterial.opacity = activeMeshViewMode === 'wireframe' ? opacity : Math.min(0.82, Math.max(0.35, opacity));
      keyLight.visible = activeRenderMode === 'mesh' && shaded;
      fillLight.visible = keyLight.visible;
      refreshLightEditing();
    };
    const clearMesh = () => {
      meshSurface?.geometry.dispose();
      meshGroup.clear();
      meshSurface = null;
      meshWireframe = null;
      loadedMeshTexture?.dispose();
      loadedMeshTexture = null;
      loadedMeshBitmap?.close();
      loadedMeshBitmap = null;
      loadedMeshHasVertexColors = false;
      meshTextureReady = true;
      meshMaterial.map = null;
      meshMaterial.vertexColors = false;
      shadedMeshMaterial.vertexColors = false;
      meshMaterial.needsUpdate = true;
      shadedMeshMaterial.needsUpdate = true;
      if (pendingTextureUrl) URL.revokeObjectURL(pendingTextureUrl);
      pendingTextureUrl = null;
      updateMeshAppearance();
      requestRender();
    };
    setMesh = (nextMesh) => {
      meshLoadGeneration += 1;
      const generation = meshLoadGeneration;
      clearMesh();
      if (!nextMesh) return;
      meshTextureReady = false;

      const previewGeometry = new THREE.BufferGeometry();
      previewGeometry.setAttribute('position', new THREE.BufferAttribute(nextMesh.positions, 3));
      if (nextMesh.uvs) previewGeometry.setAttribute('uv', new THREE.BufferAttribute(nextMesh.uvs, 2));
      if (nextMesh.colors) {
        previewGeometry.setAttribute('color', new THREE.BufferAttribute(nextMesh.colors, 3, true));
        loadedMeshHasVertexColors = true;
      }
      previewGeometry.setIndex(new THREE.BufferAttribute(nextMesh.indices, 1));
      previewGeometry.computeVertexNormals();
      previewGeometry.computeBoundingSphere();
      if (previewGeometry.boundingSphere) {
        lightHandleRadius = THREE.MathUtils.clamp(previewGeometry.boundingSphere.radius * 1.4, 0.75, 6);
      }
      meshSurface = new THREE.Mesh(previewGeometry, meshMaterial);
      meshWireframe = new THREE.Mesh(previewGeometry, wireframeMaterial);
      meshWireframe.renderOrder = 2;
      meshGroup.add(meshSurface, meshWireframe);
      updateLightVisual();
      updateMeshAppearance();
      requestRender();

      if (!nextMesh.texture?.byteLength) {
        meshTextureReady = true;
        meshMaterial.vertexColors = loadedMeshHasVertexColors && meshColorsEnabled;
        shadedMeshMaterial.vertexColors = loadedMeshHasVertexColors && meshColorsEnabled;
        meshMaterial.color.set(meshMaterial.vertexColors ? '#ffffff' : '#a9dce8');
        shadedMeshMaterial.color.set(shadedMeshMaterial.vertexColors ? '#ffffff' : '#93c6d4');
        meshMaterial.needsUpdate = true;
        shadedMeshMaterial.needsUpdate = true;
        updateMeshAppearance();
        return;
      }

      const textureBuffer = nextMesh.texture.byteOffset === 0 && nextMesh.texture.byteLength === nextMesh.texture.buffer.byteLength
        ? nextMesh.texture.buffer as ArrayBuffer
        : nextMesh.texture.slice().buffer as ArrayBuffer;
      const textureBlob = new Blob([textureBuffer], { type: 'image/png' });
      const applyTexture = (texture: THREE.Texture, bitmap: ImageBitmap | null = null) => {
        if (generation !== meshLoadGeneration) {
          texture.dispose();
          bitmap?.close();
          return;
        }
        texture.colorSpace = THREE.SRGBColorSpace;
        if (bitmap) texture.flipY = false;
        texture.needsUpdate = true;
        loadedMeshTexture = texture;
        loadedMeshBitmap = bitmap;
        meshTextureReady = true;
        updateMeshAppearance();
        meshMaterial.map = meshColorsEnabled ? texture : null;
        meshMaterial.needsUpdate = true;
        requestRender();
      };
      const loadTextureUrl = () => {
        pendingTextureUrl = URL.createObjectURL(textureBlob);
        const textureUrl = pendingTextureUrl;
        new THREE.TextureLoader().load(
          textureUrl,
          (texture) => {
            URL.revokeObjectURL(textureUrl);
            if (pendingTextureUrl === textureUrl) pendingTextureUrl = null;
            applyTexture(texture);
          },
          undefined,
          () => {
            URL.revokeObjectURL(textureUrl);
            if (pendingTextureUrl === textureUrl) pendingTextureUrl = null;
            meshTextureReady = true;
            updateMeshAppearance();
            requestRender();
          }
        );
      };
      if ('createImageBitmap' in window) {
        // ImageBitmap ignores WebGL's flip/premultiply unpack flags. Apply the
        // OBJ texture convention during off-main-thread decode and leave sRGB
        // conversion to Three.js so atlas colors are not transformed twice.
        void createImageBitmap(textureBlob, {
          imageOrientation: 'flipY',
          premultiplyAlpha: 'none',
          colorSpaceConversion: 'none'
        })
          .then((bitmap) => applyTexture(new THREE.Texture(bitmap), bitmap))
          .catch(loadTextureUrl);
      } else {
        loadTextureUrl();
      }
    };
    const clearSplat = () => {
      splatLoadGeneration += 1;
      if (loadedSplat) {
        splatGroup.remove(loadedSplat);
        loadedSplat.dispose();
        loadedSplat = null;
      }
      splatReady = false;
      splatError = '';
      splatLoadProgress = null;
      requestRender();
    };
    setSplat = (bytes) => {
      if (bytes === appliedSplatBytes) return;
      appliedSplatBytes = bytes;
      if (!bytes) {
        clearSplat();
        return;
      }
      splatLoadGeneration += 1;
      const generation = splatLoadGeneration;
      const previousSplat = loadedSplat;
      splatError = '';
      splatLoadProgress = 0;
      const nextSplat = new SplatMesh({
        fileBytes: bytes,
        fileName: 'room-splat.preview.splat',
        editable: false,
        raycastable: false,
        lod: 'quality',
        lodAbove: 2_500_000,
        nonLod: false,
        onProgress: (event) => {
          if (generation !== splatLoadGeneration || !event.lengthComputable || event.total <= 0) return;
          splatLoadProgress = THREE.MathUtils.clamp(event.loaded / event.total, 0, 1);
        }
      });
      nextSplat.visible = false;
      nextSplat.opacity = opacity;
      nextSplat.recolor.set(showColors ? '#ffffff' : '#a9dce8');
      splatGroup.add(nextSplat);
      requestRender();
      void nextSplat.initialized.then(() => {
        if (generation !== splatLoadGeneration) {
          splatGroup.remove(nextSplat);
          nextSplat.dispose();
          return;
        }
        if (previousSplat) {
          splatGroup.remove(previousSplat);
          previousSplat.dispose();
        }
        loadedSplat = nextSplat;
        nextSplat.opacity = opacity;
        nextSplat.recolor.set(showColors ? '#ffffff' : '#a9dce8');
        nextSplat.visible = true;
        splatReady = true;
        splatLoadProgress = 1;
        if (!previousSplat) {
          const bounds = nextSplat.getBoundingBox(false);
          const sphere = bounds.getBoundingSphere(new THREE.Sphere());
          if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
            const direction = camera.position.clone().sub(controls.target).normalize();
            controls.target.copy(sphere.center);
            camera.position.copy(sphere.center).addScaledVector(direction, sphere.radius * 2.6);
            camera.near = Math.max(sphere.radius / 10_000, 0.0001);
            camera.far = Math.max(sphere.radius * 100, 100);
            controls.minDistance = Math.max(sphere.radius * 0.02, 0.001);
            controls.maxDistance = Math.max(sphere.radius * 20, 30);
            camera.updateProjectionMatrix();
            controls.update();
          }
        }
        requestRender();
      }).catch((error: unknown) => {
        splatGroup.remove(nextSplat);
        nextSplat.dispose();
        if (generation !== splatLoadGeneration) return;
        splatReady = Boolean(loadedSplat);
        splatLoadProgress = null;
        splatError = error instanceof Error ? error.message : String(error);
        requestRender();
      });
    };
    setMaterial = (size, alpha, colors) => {
      material.size = size;
      material.opacity = alpha;
      material.depthWrite = alpha >= 0.98;
      if (material.vertexColors !== colors) {
        material.vertexColors = colors;
        material.needsUpdate = true;
      }
      meshColorsEnabled = colors;
      meshMaterial.opacity = alpha;
      const meshTransparent = alpha < 1;
      if (meshMaterial.transparent !== meshTransparent) {
        meshMaterial.transparent = meshTransparent;
        meshMaterial.needsUpdate = true;
      }
      shadedMeshMaterial.opacity = alpha;
      if (shadedMeshMaterial.transparent !== meshTransparent) {
        shadedMeshMaterial.transparent = meshTransparent;
        shadedMeshMaterial.needsUpdate = true;
      }
      meshMaterial.depthWrite = true;
      shadedMeshMaterial.depthWrite = true;
      const meshMap = colors ? loadedMeshTexture : null;
      if (meshMaterial.map !== meshMap) {
        meshMaterial.map = meshMap;
        meshMaterial.needsUpdate = true;
      }
      meshMaterial.color.set(colors ? '#ffffff' : '#a9dce8');
      const useVertexColors = colors && loadedMeshHasVertexColors;
      if (meshMaterial.vertexColors !== useVertexColors) {
        meshMaterial.vertexColors = useVertexColors;
        meshMaterial.needsUpdate = true;
      }
      if (shadedMeshMaterial.vertexColors !== useVertexColors) {
        shadedMeshMaterial.vertexColors = useVertexColors;
        shadedMeshMaterial.needsUpdate = true;
      }
      shadedMeshMaterial.color.set(useVertexColors ? '#ffffff' : '#93c6d4');
      updateMeshAppearance();
      if (loadedSplat) {
        loadedSplat.opacity = alpha;
        loadedSplat.recolor.set(colors ? '#ffffff' : '#a9dce8');
      }
      requestRender();
    };
    setMeshView = (mode) => {
      activeMeshViewMode = mode;
      updateMeshAppearance();
      requestRender();
    };
    setLight = (direction, editing) => {
      const nextDirection = new THREE.Vector3(...direction);
      if (nextDirection.lengthSq() > 1e-8 && Number.isFinite(nextDirection.lengthSq())) {
        appliedLightDirection = nextDirection.normalize();
      }
      lightEditingEnabled = editing;
      if (!lightTransformControls.dragging) updateLightVisual();
      else {
        lightTarget.position.copy(controls.target);
        keyLight.position.copy(controls.target).addScaledVector(appliedLightDirection, lightHandleRadius * 3);
        keyLight.target.updateMatrixWorld();
      }
      updateMeshAppearance();
      requestRender();
    };
    setRenderMode = (mode) => {
      activeRenderMode = mode;
      cloud.visible = mode === 'points';
      meshGroup.visible = mode === 'mesh';
      splatGroup.visible = mode === 'splat';
      updateMeshAppearance();
      setPixelRatio(mode === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
      requestRender();
    };
    let appliedCameraFrames: CameraFrame[] | null = null;
    setCameraFrames = (frames, visible) => {
      frustums.visible = visible && frames.length > 0;
      if (!frustums.visible) {
        requestRender();
        return;
      }
      if (frames === appliedCameraFrames) {
        requestRender();
        return;
      }
      appliedCameraFrames = frames;
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
      requestRender();
    };
    setTransform = (transform, nextAnchor) => {
      anchor.fromArray(nextAnchor);
      cloud.position.copy(anchor).multiplyScalar(-1);
      meshGroup.position.copy(cloud.position);
      splatGroup.position.copy(cloud.position);
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
      requestRender();
    };
    setGizmo = (enabled, mode) => {
      transformControls.setMode(mode as TransformControlsMode);
      transformHelper.visible = enabled;
      if (enabled) transformControls.attach(pivotGroup);
      else transformControls.detach();
      requestRender();
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
      requestRender();
    };
    const handleDragging = (event: { value: unknown }) => {
      const dragging = Boolean(event.value);
      controls.enabled = !dragging;
      if (dragging) handleInteractionStart();
      else handleInteractionEnd();
    };
    const handleTransformCommit = () => {
      emitTransform();
      onTransformCommitted();
    };
    const emitLightDirection = () => {
      const nextDirection = lightHandle.position.clone().sub(controls.target);
      if (nextDirection.lengthSq() < 1e-8) return;
      appliedLightDirection = nextDirection.normalize();
      lightTarget.position.copy(controls.target);
      keyLight.position.copy(controls.target).addScaledVector(appliedLightDirection, lightHandleRadius * 3);
      const positions = lightDirectionGeometry.getAttribute('position') as THREE.BufferAttribute;
      positions.setXYZ(0, controls.target.x, controls.target.y, controls.target.z);
      positions.setXYZ(1, lightHandle.position.x, lightHandle.position.y, lightHandle.position.z);
      positions.needsUpdate = true;
      keyLight.target.updateMatrixWorld();
      onLightDirectionChanged([appliedLightDirection.x, appliedLightDirection.y, appliedLightDirection.z]);
      requestRender();
    };
    const handleLightCommit = () => {
      updateLightVisual();
      requestRender();
    };
    transformControls.addEventListener('objectChange', emitTransform);
    transformControls.addEventListener('dragging-changed', handleDragging);
    transformControls.addEventListener('mouseUp', handleTransformCommit);
    lightTransformControls.addEventListener('objectChange', emitLightDirection);
    lightTransformControls.addEventListener('dragging-changed', handleDragging);
    lightTransformControls.addEventListener('mouseUp', handleLightCommit);
    if (packedFrame) setPackedGeometry(packedFrame);
    else setGeometry(points);
    setMesh(mesh);
    setSplat(splatBytes);
    setMaterial(pointSize, opacity, showColors);
    setMeshView(meshViewMode);
    setLight(lightDirection, lightEditMode);
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
        requestRender();
      });
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas.parentElement!);
    resize();

    let animationFrame = 0;
    let lastAnimationTime = performance.now();
    const animate = (time = performance.now()) => {
      animationFrame = requestAnimationFrame(animate);
      const deltaSeconds = Math.min(0.05, Math.max(0.001, (time - lastAnimationTime) / 1000));
      lastAnimationTime = time;
      let controlsChanged = controls.update(deltaSeconds);
      let zoomChanged = false;
      if (smoothZoomActive) {
        const currentDistance = controls.getDistance();
        const distanceDelta = targetZoomDistance - currentDistance;
        const settleThreshold = Math.max(0.0005, targetZoomDistance * 0.0002);
        const inputStillArriving = time - lastZoomInputTime < 110;
        const nextDistance = !inputStillArriving && Math.abs(distanceDelta) <= settleThreshold
          ? targetZoomDistance
          : THREE.MathUtils.damp(currentDistance, targetZoomDistance, inputStillArriving ? 6 : 13, deltaSeconds);
        const direction = camera.position.clone().sub(controls.target);
        if (direction.lengthSq() > 1e-10) {
          camera.position.copy(controls.target).addScaledVector(direction.normalize(), nextDistance);
          controlsChanged = controls.update(deltaSeconds) || controlsChanged;
          zoomChanged = true;
        }
        if (!inputStillArriving && nextDistance === targetZoomDistance) {
          smoothZoomActive = false;
          handleInteractionEnd();
        }
      }
      if (!controlsChanged && !zoomChanged && !renderRequested) return;
      renderRequested = false;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      cancelAnimationFrame(resizeFrame);
      window.clearInterval(loadingElapsedTimer);
      window.clearTimeout(restoreQualityTimer);
      canvas.removeEventListener('pointerup', handlePointer);
      canvas.removeEventListener('wheel', handleSmoothWheel, true);
      resizeObserver.disconnect();
      controls.removeEventListener('change', handleControlsChange);
      controls.removeEventListener('start', handleInteractionStart);
      controls.removeEventListener('end', handleInteractionEnd);
      controls.dispose();
      transformControls.removeEventListener('objectChange', emitTransform);
      transformControls.removeEventListener('dragging-changed', handleDragging);
      transformControls.removeEventListener('mouseUp', handleTransformCommit);
      transformControls.detach();
      transformControls.dispose();
      scene.remove(transformHelper);
      meshLoadGeneration += 1;
      clearMesh();
      lightTransformControls.removeEventListener('objectChange', emitLightDirection);
      lightTransformControls.removeEventListener('dragging-changed', handleDragging);
      lightTransformControls.removeEventListener('mouseUp', handleLightCommit);
      lightTransformControls.detach();
      lightTransformControls.dispose();
      scene.remove(lightTransformHelper);
      clearSplat();
      sparkRenderer.dispose();
      meshMaterial.dispose();
      shadedMeshMaterial.dispose();
      wireframeMaterial.dispose();
      lightHandleGeometry.dispose();
      lightHandleMaterial.dispose();
      lightDirectionGeometry.dispose();
      lightDirectionMaterial.dispose();
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
    {processing ? renderMode === 'splat' ? 'Training splats · live preview' : 'Reconstructing points · live preview' : live ? 'Live sensor point cloud' : renderMode === 'splat' ? splatError ? 'Splat preview failed' : splatReady ? 'Gaussian splat view' : 'Loading Gaussian splat' : renderMode === 'mesh' && mesh ? meshViewMode === 'surface-wireframe' ? 'Mesh + wireframe view' : meshViewMode === 'wireframe' ? 'Wireframe view' : meshViewMode === 'shaded' ? 'Shaded mesh view' : 'Textured mesh view' : points.length ? 'Point-cloud view' : 'Awaiting sensor'}
  </div>
  {#if floorPickMode}
    <div class="viewer-hud floor-hint">Click a dense patch of floor</div>
  {:else if anchorPickMode}
    <div class="viewer-hud floor-hint">Click the model to place the gizmo anchor</div>
  {:else if lightEditMode && renderMode === 'mesh' && meshViewMode === 'shaded'}
    <div class="viewer-hud floor-hint">Drag the light handle to change its direction</div>
  {/if}
  {#if visibleAssetLoading}
    <div class="asset-loading" aria-live="polite" aria-busy="true">
      <div class="asset-loading-spinner"></div>
      <strong>{visibleAssetLoading === 'points' ? 'Loading point-cloud preview' : visibleAssetLoading === 'mesh' ? 'Loading mesh geometry' : visibleAssetLoading === 'mesh-texture' ? 'Decoding preview texture' : visibleAssetLoading === 'splat' ? 'Loading Gaussian data' : 'Preparing Gaussian renderer'}</strong>
      <span>{visibleAssetLoading === 'points' ? 'Reading the optimized point sample' : visibleAssetLoading === 'mesh' ? 'Reading every mesh vertex and triangle' : visibleAssetLoading === 'mesh-texture' ? 'Decoding and uploading the full-resolution texture' : visibleAssetLoading === 'splat' ? 'Reading the compact preview from disk' : 'Uploading and sorting splats on the GPU'}{visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null ? ` · ${Math.round(splatLoadProgress * 100)}%` : ''} · {assetLoadingElapsed}s</span>
      <div class:determinate={visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null} class="asset-loading-track"><i style={visibleAssetLoading === 'splat-gpu' && splatLoadProgress !== null ? `width: ${Math.max(2, splatLoadProgress * 100)}%` : undefined}></i></div>
    </div>
  {/if}
  {#if renderMode === 'splat' ? splatReady : renderMode === 'mesh' ? mesh : packedFrame ? packedFrame.pointCount > 0 : points.length > 0}
    <div class="viewer-hud bottom-right">Drag to orbit · Scroll to zoom{showCameraFrames && cameraFrames.length ? ` · ${cameraFrames.length} camera poses` : ''}</div>
  {:else}
    <div class="empty-state">
      <strong>{processing ? renderMode === 'splat' ? 'Preparing the next splat snapshot…' : 'Preparing reconstruction geometry…' : renderMode === 'splat' ? splatError || (splatBytes ? 'Loading Gaussian splat…' : 'No Gaussian splat yet') : renderMode === 'mesh' ? 'No reconstructed mesh yet' : 'No live depth points yet'}</strong>
      <span>{processing ? renderMode === 'splat' ? 'The viewer updates periodically as Gaussian optimization continues.' : 'The viewer updates whenever the reconstruction worker publishes usable geometry.' : renderMode === 'splat' ? splatError ? 'Rebuild or export the PLY to inspect the generated artifact.' : splatBytes ? 'Decoding and uploading Gaussian covariance data to the GPU.' : 'Build the Gaussian splat artifact to visualize it here.' : renderMode === 'mesh' ? 'Build the 3D model to create a textured mesh.' : 'The preview starts when the selected sensor streams.'}</span>
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
  .asset-loading { position: absolute; z-index: 5; left: 50%; top: 50%; width: min(340px, calc(100% - 40px)); padding: 22px 24px; display: grid; justify-items: center; gap: 9px; transform: translate(-50%, -50%); border: 1px solid rgba(112, 186, 215, 0.28); border-radius: 16px; background: rgba(5, 14, 23, 0.94); box-shadow: 0 22px 70px rgba(0, 0, 0, 0.42); color: #b7ccd7; text-align: center; pointer-events: none; }
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
