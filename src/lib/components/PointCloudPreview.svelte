<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import type { SparkRenderer as SparkRendererInstance, SplatMesh as SplatMeshInstance } from '@sparkjsdev/spark';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import type { TransformControlsMode } from 'three/examples/jsm/controls/TransformControls.js';
  import { AnchorAngleTransformControls } from '../controls/AnchorAngleTransformControls';
  import type { BoundingBoxClip, CloudTransform, MeshViewMode, PackedPreviewFrame, PreviewMesh, PreviewPoint } from '../types';

  type RenderMode = 'points' | 'mesh' | 'splat';
  type AssetLoadingState = 'points' | 'mesh' | 'mesh-texture' | 'splat' | 'splat-gpu' | null;
  type ProjectionMode = 'orthographic' | 'perspective';

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
  export let splatRepresentation: '2d' | '3d' = '2d';
  export let assetLoading: 'points' | 'mesh' | 'splat' | null = null;
  export let floorPickMode = false;
  export let cloudTransform: CloudTransform = { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] };
  export let gizmoAnchor: [number, number, number] = [0, 0, 0];
  export let editMode = false;
  export let gizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
  export let rotationSnapDegrees = 0;
  export let clipBounds: BoundingBoxClip | null = null;
  export let clipEditMode = false;
  export let clipGizmoMode: 'translate' | 'scale' = 'scale';
  export let onFloorDetected: (transform: CloudTransform) => void = () => undefined;
  export let onFloorMessage: (message: string) => void = () => undefined;
  export let onTransformChanged: (transform: CloudTransform) => void = () => undefined;
  export let onTransformCommitted: () => void = () => undefined;
  export let onClipBoundsChanged: (bounds: BoundingBoxClip) => void = () => undefined;
  export let onClipBoundsCommitted: () => void = () => undefined;

  let canvas: HTMLCanvasElement;
  let viewCubeCanvas: HTMLCanvasElement;
  let setPoints: (next: PreviewPoint[]) => void = () => undefined;
  let setPackedPoints: (next: PackedPreviewFrame) => void = () => undefined;
  let setMaterial: (size: number, alpha: number, colors: boolean) => void = () => undefined;
  let setMesh: (next: PreviewMesh | null) => void = () => undefined;
  let setMeshMode: (next: MeshViewMode) => void = () => undefined;
  let setSplat: (next: Uint8Array | null, force?: boolean) => void = () => undefined;
  let setRenderMode: (next: RenderMode) => void = () => undefined;
  let setTransform: (transform: CloudTransform, anchor: [number, number, number]) => void = () => undefined;
  let setClip: (bounds: BoundingBoxClip | null) => void = () => undefined;
  let setGizmo: (modelEnabled: boolean, modelMode: 'translate' | 'rotate' | 'scale', clipEnabled: boolean, clipMode: 'translate' | 'scale') => void = () => undefined;
  let setRotationSnap: (degrees: number) => void = () => undefined;
  let resetCameraView: () => void = () => undefined;
  let toggleProjection: () => void = () => undefined;
  let projectionMode: ProjectionMode = 'perspective';
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
  $: setClip(clipBounds);
  $: setSplat(splatBytes);
  $: setRenderMode(renderMode);
  $: setTransform(cloudTransform, gizmoAnchor);
  $: setGizmo(editMode, gizmoMode, clipEditMode, clipGizmoMode);
  $: setRotationSnap(rotationSnapDegrees);

  function clippedSplatBytes(bytes: Uint8Array, bounds: BoundingBoxClip | null, transform: CloudTransform): Uint8Array {
    if (!bounds || bytes.byteLength % 32 !== 0) return bytes;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const output = new Uint8Array(bytes.byteLength);
    const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
      THREE.MathUtils.degToRad(transform.rotation[0]),
      THREE.MathUtils.degToRad(transform.rotation[1]),
      THREE.MathUtils.degToRad(transform.rotation[2]),
      'XYZ'
    ));
    const elements = new THREE.Matrix4().compose(
      new THREE.Vector3().fromArray(transform.position),
      rotation,
      new THREE.Vector3().fromArray(transform.scale)
    ).elements;
    let outputOffset = 0;
    for (let offset = 0; offset < bytes.byteLength; offset += 32) {
      const x = view.getFloat32(offset, true);
      const y = view.getFloat32(offset + 4, true);
      const z = view.getFloat32(offset + 8, true);
      const transformed = [
        elements[0] * x + elements[4] * y + elements[8] * z + elements[12],
        elements[1] * x + elements[5] * y + elements[9] * z + elements[13],
        elements[2] * x + elements[6] * y + elements[10] * z + elements[14]
      ];
      const inside = transformed.every((value, axis) => Number.isFinite(value) && value >= bounds.min[axis] && value <= bounds.max[axis]);
      if (!inside) continue;
      output.set(bytes.subarray(offset, offset + 32), outputOffset);
      outputOffset += 32;
    }
    if (outputOffset === bytes.byteLength) return bytes;
    return output.slice(0, outputOffset);
  }

  onMount(() => {
    const loadingTimer = window.setInterval(() => {
      if (loadingSince) loadingElapsed = Math.floor((performance.now() - loadingSince) / 1000);
    }, 250);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b0d10');
    scene.fog = new THREE.FogExp2('#0b0d10', 0.055);

    const perspectiveCamera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    perspectiveCamera.position.set(6.8, 4.7, 7.6);
    orthographicCamera.position.copy(perspectiveCamera.position);
    let camera: THREE.Camera = perspectiveCamera;
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance'
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.localClippingEnabled = true;

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
    let cameraTween: {
      startedAt: number;
      fromOffset: THREE.Vector3;
      rotation: THREE.Quaternion;
      projectionAfter: ProjectionMode | null;
    } | null = null;
    let applyProjectionMode: (next: ProjectionMode) => void = () => undefined;
    let controlsStartDirection: THREE.Vector3 | null = null;
    let restoreQualityTimer = 0;
    const interactionStart = () => {
      cameraTween = null;
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
    const handleControlsChange = () => {
      if (projectionMode === 'orthographic' && controlsStartDirection && !cameraTween) {
        const direction = camera.position.clone().sub(controls.target).normalize();
        if (direction.angleTo(controlsStartDirection) > 0.001) applyProjectionMode('perspective');
      }
      invalidate();
    };
    const handleControlsStart = () => {
      controlsStartDirection = camera.position.clone().sub(controls.target).normalize();
      interactionStart();
    };
    const handleControlsEnd = () => {
      controlsStartDirection = null;
      interactionEnd();
    };
    controls.addEventListener('change', handleControlsChange);
    controls.addEventListener('start', handleControlsStart);
    controls.addEventListener('end', handleControlsEnd);

    const viewCubeScene = new THREE.Scene();
    const viewCubeOrthographicCamera = new THREE.OrthographicCamera(-2.05, 2.05, 2.05, -2.05, 0.1, 20);
    const viewCubePerspectiveCamera = new THREE.PerspectiveCamera(34, 1, 0.1, 20);
    viewCubeOrthographicCamera.position.set(0, 0, 6);
    viewCubePerspectiveCamera.position.copy(viewCubeOrthographicCamera.position);
    let viewCubeCamera: THREE.Camera = viewCubePerspectiveCamera;
    const viewCubeRenderer = new THREE.WebGLRenderer({
      canvas: viewCubeCanvas,
      alpha: true,
      antialias: true,
      powerPreference: 'low-power'
    });
    viewCubeRenderer.setClearColor(0x000000, 0);
    viewCubeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    viewCubeRenderer.setSize(118, 118, false);
    viewCubeRenderer.outputColorSpace = THREE.SRGBColorSpace;

    const makeFaceTexture = (label: string, axis: string, color: string) => {
      const textureCanvas = document.createElement('canvas');
      textureCanvas.width = 256;
      textureCanvas.height = 256;
      const context = textureCanvas.getContext('2d')!;
      context.fillStyle = color;
      context.fillRect(0, 0, 256, 256);
      context.strokeStyle = 'rgba(172, 229, 247, 0.5)';
      context.lineWidth = 10;
      context.strokeRect(5, 5, 246, 246);
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillStyle = '#f0fbfe';
      context.font = '850 58px Inter, Segoe UI, sans-serif';
      context.fillText(axis, 128, 112);
      context.fillStyle = 'rgba(232, 248, 252, 0.9)';
      context.font = `800 ${label.length > 5 ? 22 : 25}px Inter, Segoe UI, sans-serif`;
      context.fillText(label, 128, 171);
      const texture = new THREE.CanvasTexture(textureCanvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = Math.min(4, viewCubeRenderer.capabilities.getMaxAnisotropy());
      return texture;
    };

    const viewCubeFaceSpecs = [
      ['RIGHT', '+X', '#b64f59', '#6e2833'],
      ['LEFT', '-X', '#923e49', '#511e29'],
      ['TOP', '+Y', '#4f9b6d', '#245c3d'],
      ['BOTTOM', '-Y', '#397952', '#193f2c'],
      ['BACK', '-Z', '#306797', '#17385f'],
      ['FRONT', '+Z', '#397fb5', '#1e4c78']
    ] as const;
    const viewCubeTextures = viewCubeFaceSpecs.map(([label, axis, color]) =>
      makeFaceTexture(label, axis, color)
    );
    const viewCubeMaterials = viewCubeTextures.map((map) => new THREE.MeshBasicMaterial({ map }));
    const viewCubeGeometry = new THREE.BoxGeometry(1.42, 1.42, 1.42);
    const viewCubeMesh = new THREE.Mesh(viewCubeGeometry, viewCubeMaterials);
    const viewCubeRoot = new THREE.Group();
    viewCubeRoot.add(viewCubeMesh);
    const viewCubeEdgeGeometry = new THREE.EdgesGeometry(viewCubeGeometry);
    const viewCubeEdgeMaterial = new THREE.LineBasicMaterial({
      color: '#d0edf4',
      transparent: true,
      opacity: 0.8
    });
    const viewCubeEdges = new THREE.LineSegments(viewCubeEdgeGeometry, viewCubeEdgeMaterial);
    viewCubeEdges.scale.setScalar(1.003);
    viewCubeRoot.add(viewCubeEdges);
    viewCubeScene.add(viewCubeRoot);

    const viewCubeRaycaster = new THREE.Raycaster();
    const viewCubePointer = new THREE.Vector2();
    const pickViewCubeFace = (event: PointerEvent) => {
      const bounds = viewCubeCanvas.getBoundingClientRect();
      viewCubePointer.set(
        ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
        -((event.clientY - bounds.top) / bounds.height) * 2 + 1
      );
      viewCubeRoot.updateMatrixWorld(true);
      viewCubeRaycaster.setFromCamera(viewCubePointer, viewCubeCamera);
      return viewCubeRaycaster.intersectObject(viewCubeMesh, false)[0] ?? null;
    };
    const updateViewCubeHover = (materialIndex: number | null) => {
      viewCubeMaterials.forEach((material, index) => {
        material.color.set(index === materialIndex ? '#bfefff' : '#ffffff');
      });
      invalidate();
    };
    const snapCameraTo = (direction: THREE.Vector3, projectionAfter: ProjectionMode | null = null) => {
      if (projectionAfter === 'perspective') applyProjectionMode('perspective');
      const fromOffset = camera.position.clone().sub(controls.target);
      if (fromOffset.lengthSq() < 1e-8 || direction.lengthSq() < 1e-8) return;
      const fromDirection = fromOffset.clone().normalize();
      const toDirection = direction.clone().normalize();
      cameraTween = {
        startedAt: performance.now(),
        fromOffset,
        rotation: new THREE.Quaternion().setFromUnitVectors(fromDirection, toDirection),
        projectionAfter
      };
      window.clearTimeout(restoreQualityTimer);
      applyPixelRatio(interactionPixelRatio);
      invalidate();
    };
    resetCameraView = () => snapCameraTo(new THREE.Vector3(0.68, 0.48, 0.76), 'perspective');

    let viewCubePointerId: number | null = null;
    let viewCubeDragging = false;
    let viewCubeLastX = 0;
    let viewCubeLastY = 0;
    let viewCubeTravel = 0;
    const handleViewCubePointerDown = (event: PointerEvent) => {
      event.preventDefault();
      viewCubePointerId = event.pointerId;
      viewCubeDragging = false;
      viewCubeTravel = 0;
      viewCubeLastX = event.clientX;
      viewCubeLastY = event.clientY;
      viewCubeCanvas.setPointerCapture(event.pointerId);
      interactionStart();
    };
    const handleViewCubePointerMove = (event: PointerEvent) => {
      if (viewCubePointerId !== event.pointerId) {
        const hit = pickViewCubeFace(event);
        updateViewCubeHover(hit?.face?.materialIndex ?? null);
        return;
      }
      const deltaX = event.clientX - viewCubeLastX;
      const deltaY = event.clientY - viewCubeLastY;
      viewCubeLastX = event.clientX;
      viewCubeLastY = event.clientY;
      viewCubeTravel += Math.abs(deltaX) + Math.abs(deltaY);
      if (viewCubeTravel > 4 && !viewCubeDragging) {
        viewCubeDragging = true;
        applyProjectionMode('perspective');
      }
      if (!viewCubeDragging) return;

      const offset = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= deltaX * 0.012;
      spherical.phi = THREE.MathUtils.clamp(spherical.phi - deltaY * 0.012, 0.025, Math.PI - 0.025);
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
      camera.lookAt(controls.target);
      controls.update();
      invalidate();
    };
    const handleViewCubePointerUp = (event: PointerEvent) => {
      if (viewCubePointerId !== event.pointerId) return;
      if (viewCubeCanvas.hasPointerCapture(event.pointerId)) viewCubeCanvas.releasePointerCapture(event.pointerId);
      viewCubePointerId = null;
      if (viewCubeDragging) {
        viewCubeDragging = false;
        interactionEnd();
        return;
      }
      const hit = pickViewCubeFace(event);
      if (!hit?.face) {
        interactionEnd();
        return;
      }
      const localPoint = viewCubeMesh.worldToLocal(hit.point.clone());
      const direction = hit.face.normal.clone();
      const edgeThreshold = 0.51;
      if (Math.abs(localPoint.x) > edgeThreshold) direction.x = Math.sign(localPoint.x);
      if (Math.abs(localPoint.y) > edgeThreshold) direction.y = Math.sign(localPoint.y);
      if (Math.abs(localPoint.z) > edgeThreshold) direction.z = Math.sign(localPoint.z);
      const componentCount = [direction.x, direction.y, direction.z]
        .filter((component) => Math.abs(component) > 0.5).length;
      snapCameraTo(direction.normalize(), componentCount === 1 ? 'orthographic' : 'perspective');
    };
    const handleViewCubePointerLeave = () => {
      if (viewCubePointerId === null) updateViewCubeHover(null);
    };
    viewCubeCanvas.addEventListener('pointerdown', handleViewCubePointerDown);
    viewCubeCanvas.addEventListener('pointermove', handleViewCubePointerMove);
    viewCubeCanvas.addEventListener('pointerup', handleViewCubePointerUp);
    viewCubeCanvas.addEventListener('pointercancel', handleViewCubePointerUp);
    viewCubeCanvas.addEventListener('pointerleave', handleViewCubePointerLeave);

    const grid = new THREE.GridHelper(12, 24, '#30343a', '#202328');
    grid.position.y = -0.015;
    scene.add(grid);
    const fillLight = new THREE.HemisphereLight('#d9f3ff', '#17232b', 1.15);
    const keyLight = new THREE.DirectionalLight('#fff1d6', 3.2);
    keyLight.position.set(3, 6, 4);
    scene.add(fillLight, keyLight);

    const root = new THREE.Group();
    scene.add(root);
    const anchor = new THREE.Vector3();

    const clipBoxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const clipBoxMaterial = new THREE.MeshBasicMaterial({
      color: '#efb366',
      transparent: true,
      opacity: 0.07,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    const clipBox = new THREE.Mesh(clipBoxGeometry, clipBoxMaterial);
    const clipEdgeGeometry = new THREE.EdgesGeometry(clipBoxGeometry);
    const clipEdgeMaterial = new THREE.LineBasicMaterial({ color: '#efb366', transparent: true, opacity: 0.92 });
    clipBox.add(new THREE.LineSegments(clipEdgeGeometry, clipEdgeMaterial));
    clipBox.visible = false;
    clipBox.renderOrder = 10;
    scene.add(clipBox);

    const clippingPlanes = Array.from({ length: 6 }, () => new THREE.Plane());
    let activeClipBounds: BoundingBoxClip | null = null;
    let appliedClipKey = 'off';
    let activeCloudTransform: CloudTransform = cloudTransform;
    let appliedTransformKey = '';

    const transformControls = new AnchorAngleTransformControls(camera, renderer.domElement);
    const transformHelper = transformControls.getHelper();
    transformControls.setSpace('world');
    transformControls.setSize(0.82);
    transformControls.translationSnap = 0.01;
    transformControls.rotationSnap = null;
    transformControls.scaleSnap = 0.01;
    transformHelper.visible = false;
    scene.add(transformHelper);

    const viewportAspect = () => {
      const bounds = canvas.getBoundingClientRect();
      return bounds.height > 0 ? Math.max(0.01, bounds.width / bounds.height) : perspectiveCamera.aspect;
    };
    applyProjectionMode = (next) => {
      if (next === projectionMode) return;
      const aspect = viewportAspect();
      if (next === 'orthographic') {
        const distance = Math.max(0.01, camera.position.distanceTo(controls.target));
        const verticalSpan = 2 * distance * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov * 0.5));
        orthographicCamera.left = -verticalSpan * aspect * 0.5;
        orthographicCamera.right = verticalSpan * aspect * 0.5;
        orthographicCamera.top = verticalSpan * 0.5;
        orthographicCamera.bottom = -verticalSpan * 0.5;
        orthographicCamera.zoom = 1;
        orthographicCamera.position.copy(camera.position);
        orthographicCamera.quaternion.copy(camera.quaternion);
        orthographicCamera.up.copy(camera.up);
        orthographicCamera.updateProjectionMatrix();
        camera = orthographicCamera;
        viewCubeCamera = viewCubeOrthographicCamera;
      } else {
        if (camera === orthographicCamera) {
          const verticalSpan = (orthographicCamera.top - orthographicCamera.bottom) / orthographicCamera.zoom;
          const distance = verticalSpan
            / (2 * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov * 0.5)));
          const direction = orthographicCamera.position.clone().sub(controls.target).normalize();
          perspectiveCamera.position.copy(controls.target).addScaledVector(direction, distance);
          perspectiveCamera.quaternion.copy(orthographicCamera.quaternion);
          perspectiveCamera.up.copy(orthographicCamera.up);
        }
        perspectiveCamera.aspect = aspect;
        perspectiveCamera.updateProjectionMatrix();
        camera = perspectiveCamera;
        viewCubeCamera = viewCubePerspectiveCamera;
      }
      controls.object = camera;
      transformControls.camera = camera;
      projectionMode = next;
      invalidate();
    };
    toggleProjection = () => {
      applyProjectionMode(projectionMode === 'perspective' ? 'orthographic' : 'perspective');
    };

    const pointGeometry = new THREE.BufferGeometry();
    const pointMaterial = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: showColors,
      color: '#a9dce8',
      sizeAttenuation: true,
      transparent: opacity < 1,
      depthWrite: opacity >= 0.98,
      opacity,
      clippingPlanes: null
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
      polygonOffsetUnits: 1,
      clippingPlanes: null
    });
    const shadedMaterial = new THREE.MeshStandardMaterial({
      color: '#93c6d4',
      roughness: 0.78,
      metalness: 0.02,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      clippingPlanes: null
    });
    const wireMaterial = new THREE.MeshBasicMaterial({
      color: '#69cbea',
      side: THREE.DoubleSide,
      wireframe: true,
      transparent: true,
      depthWrite: false,
      opacity,
      clippingPlanes: null
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

    const updateClippingPlanes = (bounds: BoundingBoxClip | null) => {
      const materials = [pointMaterial, surfaceMaterial, shadedMaterial, wireMaterial];
      if (!bounds) {
        for (const material of materials) {
          if (material.clippingPlanes !== null) {
            material.clippingPlanes = null;
            material.needsUpdate = true;
          }
        }
        invalidate();
        return;
      }
      // Three.js discards the negative half-space of each material clipping
      // plane. Keep the box interior by pointing the minimum planes inward
      // and the maximum planes inward from the opposite side. These planes
      // are expressed in world space, so clipping remains after `root`'s
      // model transform, matching the CPU-filtered splat path.
      clippingPlanes[0].setComponents(1, 0, 0, -bounds.min[0]);
      clippingPlanes[1].setComponents(-1, 0, 0, bounds.max[0]);
      clippingPlanes[2].setComponents(0, 1, 0, -bounds.min[1]);
      clippingPlanes[3].setComponents(0, -1, 0, bounds.max[1]);
      clippingPlanes[4].setComponents(0, 0, 1, -bounds.min[2]);
      clippingPlanes[5].setComponents(0, 0, -1, bounds.max[2]);
      for (const material of materials) {
        if (material.clippingPlanes !== clippingPlanes) {
          material.clippingPlanes = clippingPlanes;
          material.needsUpdate = true;
        }
      }
      invalidate();
    };

    const boundsFromClipBox = (): BoundingBoxClip => {
      const halfSize = new THREE.Vector3(
        Math.max(0.001, Math.abs(clipBox.scale.x)) * 0.5,
        Math.max(0.001, Math.abs(clipBox.scale.y)) * 0.5,
        Math.max(0.001, Math.abs(clipBox.scale.z)) * 0.5
      );
      return {
        min: [clipBox.position.x - halfSize.x, clipBox.position.y - halfSize.y, clipBox.position.z - halfSize.z],
        max: [clipBox.position.x + halfSize.x, clipBox.position.y + halfSize.y, clipBox.position.z + halfSize.z]
      };
    };

    setClip = (bounds) => {
      const key = bounds ? [...bounds.min, ...bounds.max].join(':') : 'off';
      if (key === appliedClipKey) return;
      appliedClipKey = key;
      activeClipBounds = bounds
        ? { min: [...bounds.min], max: [...bounds.max] }
        : null;
      clipBox.visible = Boolean(bounds);
      if (bounds) {
        clipBox.position.set(
          (bounds.min[0] + bounds.max[0]) * 0.5,
          (bounds.min[1] + bounds.max[1]) * 0.5,
          (bounds.min[2] + bounds.max[2]) * 0.5
        );
        clipBox.rotation.set(0, 0, 0);
        clipBox.scale.set(
          Math.max(0.001, bounds.max[0] - bounds.min[0]),
          Math.max(0.001, bounds.max[1] - bounds.min[1]),
          Math.max(0.001, bounds.max[2] - bounds.min[2])
        );
      }
      updateClippingPlanes(activeClipBounds);
      if (appliedSplat) setSplat(appliedSplat, true);
    };

    const splatGroup = new THREE.Group();
    root.add(splatGroup);
    let loadedSplat: SplatMeshInstance | null = null;
    let appliedSplat: Uint8Array | null = null;
    let appliedSplatView: DataView | null = null;
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
    setSplat = (next, force = false) => {
      if (!force && next === appliedSplat) return;
      appliedSplat = next;
      appliedSplatView = next
        ? new DataView(next.buffer, next.byteOffset, next.byteLength)
        : null;
      if (!next) {
        clearSplat();
        return;
      }
      const displayBytes = clippedSplatBytes(next, activeClipBounds, activeCloudTransform);
      splatGeneration += 1;
      const generation = splatGeneration;
      splatReady = false;
      splatError = '';
      splatLoadProgress = 0;
      if (!displayBytes.byteLength) {
        if (loadedSplat) {
          splatGroup.remove(loadedSplat);
          loadedSplat.dispose();
          loadedSplat = null;
        }
        splatLoadProgress = null;
        splatError = 'Bounding box contains no Gaussian centers';
        invalidate();
        return;
      }
      void ensureSpark().then((module) => {
        if (!module || generation !== splatGeneration) return;
        const previous = loadedSplat;
        const candidate = new module.SplatMesh({
          fileBytes: displayBytes,
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
      grid.visible = true;
      applyPixelRatio(next === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
      updateMeshAppearance();
      invalidate();
    };

    setTransform = (transform, nextAnchor) => {
      const transformKey = [...transform.position, ...transform.rotation, ...transform.scale].join(':');
      const transformChanged = transformKey !== appliedTransformKey;
      appliedTransformKey = transformKey;
      activeCloudTransform = {
        position: [...transform.position],
        rotation: [...transform.rotation],
        scale: [...transform.scale]
      };
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
      if (transformChanged && appliedSplat) setSplat(appliedSplat, true);
      invalidate();
    };
    let activeGizmoMode: 'translate' | 'rotate' | 'scale' = 'translate';
    let gizmoTarget: 'model' | 'clip' | null = null;
    setGizmo = (modelEnabled, modelMode, clipEnabled, clipMode) => {
      const nextTarget = clipEnabled && activeClipBounds ? 'clip' : modelEnabled ? 'model' : null;
      const nextMode = nextTarget === 'clip' ? clipMode : modelMode;
      if (nextMode !== activeGizmoMode) {
        activeGizmoMode = nextMode;
        transformControls.setMode(nextMode as TransformControlsMode);
      }
      if (nextTarget !== gizmoTarget) {
        gizmoTarget = nextTarget;
        transformControls.detach();
        transformHelper.visible = Boolean(nextTarget);
        if (nextTarget) {
          transformControls.setSpace('world');
          transformControls.attach(nextTarget === 'clip' ? clipBox : root);
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
      else if (!editMode && !clipEditMode) interactionEnd();
    };
    const commitTransform = () => {
      if (gizmoTarget === 'clip') {
        const bounds = boundsFromClipBox();
        activeClipBounds = bounds;
        updateClippingPlanes(bounds);
        onClipBoundsChanged(bounds);
        onClipBoundsCommitted();
        return;
      }
      emitTransform();
      onTransformCommitted();
    };
    const handleGizmoObjectChange = () => {
      if (gizmoTarget === 'clip') updateClippingPlanes(boundsFromClipBox());
      invalidate();
    };
    // TransformControls already edits `root` directly. Sending every pointer
    // event through Svelte only writes the same matrix back and makes large
    // scenes feel CPU-bound. Publish the final pose once on mouse-up instead.
    transformControls.addEventListener('objectChange', handleGizmoObjectChange);
    transformControls.addEventListener('dragging-changed', handleGizmoDragging);
    transformControls.addEventListener('mouseUp', commitTransform);

    const sourcePointCount = () => activeRenderMode === 'splat'
      ? Math.floor((appliedSplat?.byteLength ?? 0) / 32)
      : points.length || Math.floor((mesh?.positions.length ?? 0) / 3);
    const sourcePoint = (index: number, target = new THREE.Vector3()) => {
      if (activeRenderMode === 'splat' && appliedSplatView) {
        const offset = index * 32;
        return target.set(
          appliedSplatView.getFloat32(offset, true),
          appliedSplatView.getFloat32(offset + 4, true),
          appliedSplatView.getFloat32(offset + 8, true)
        );
      }
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
      if (!floorPickMode || sourcePointCount() === 0) return;
      const bounds = canvas.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const contentObject = activeRenderMode === 'mesh'
        ? meshGroup
        : activeRenderMode === 'splat' ? splatGroup : pointCloud;
      if (activeRenderMode === 'splat') {
        root.updateMatrixWorld(true);
        const count = sourcePointCount();
        const stride = Math.max(1, Math.floor(count / 200_000));
        const localPoint = new THREE.Vector3();
        const worldPoint = new THREE.Vector3();
        const cameraOffset = new THREE.Vector3();
        let selected: THREE.Vector3 | null = null;
        let bestAngularDistance = 0.0002;
        for (let index = 0; index < count; index += stride) {
          sourcePoint(index, localPoint);
          worldPoint.copy(localPoint).applyMatrix4(contentObject.matrixWorld);
          cameraOffset.copy(worldPoint).sub(raycaster.ray.origin);
          const forwardDistance = raycaster.ray.direction.dot(cameraOffset);
          if (forwardDistance <= 0) continue;
          const angularDistance = raycaster.ray.distanceSqToPoint(worldPoint) / (forwardDistance * forwardDistance);
          if (angularDistance < bestAngularDistance) {
            bestAngularDistance = angularDistance;
            selected = localPoint.clone();
          }
        }
        if (!selected) {
          onFloorMessage('No Gaussian surface selected. Click directly on a dense floor patch.');
          return;
        }
        fitFloor(selected, contentObject);
        return;
      }
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
    setMaterial(pointSize, opacity, showColors);
    setMeshMode(meshViewMode);
    setRenderMode(renderMode);
    setTransform(cloudTransform, gizmoAnchor);
    setClip(clipBounds);
    setSplat(splatBytes);
    setGizmo(editMode, gizmoMode, clipEditMode, clipGizmoMode);
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
        const aspect = width / height;
        perspectiveCamera.aspect = aspect;
        perspectiveCamera.updateProjectionMatrix();
        const orthographicHalfHeight = (orthographicCamera.top - orthographicCamera.bottom) * 0.5;
        orthographicCamera.left = -orthographicHalfHeight * aspect;
        orthographicCamera.right = orthographicHalfHeight * aspect;
        orthographicCamera.updateProjectionMatrix();
        invalidate();
      });
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas.parentElement!);
    resize();

    let animationFrame = 0;
    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      let tweenChanged = false;
      if (cameraTween) {
        const elapsed = (performance.now() - cameraTween.startedAt) / 360;
        const progress = THREE.MathUtils.clamp(elapsed, 0, 1);
        const eased = progress < 0.5
          ? 4 * progress * progress * progress
          : 1 - Math.pow(-2 * progress + 2, 3) / 2;
        const stepRotation = new THREE.Quaternion().slerp(cameraTween.rotation, eased);
        const offset = cameraTween.fromOffset.clone().applyQuaternion(stepRotation);
        camera.position.copy(controls.target).add(offset);
        camera.lookAt(controls.target);
        tweenChanged = true;
        if (progress >= 1) {
          const projectionAfter = cameraTween.projectionAfter;
          cameraTween = null;
          if (projectionAfter) applyProjectionMode(projectionAfter);
          interactionEnd();
        }
      }
      const changed = controls.update();
      if (!changed && !tweenChanged && !renderInvalidated) return;
      renderInvalidated = false;
      renderer.render(scene, camera);
      viewCubeRoot.quaternion.copy(camera.quaternion).invert();
      viewCubeRenderer.render(viewCubeScene, viewCubeCamera);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      cancelAnimationFrame(resizeFrame);
      window.clearInterval(loadingTimer);
      window.clearTimeout(restoreQualityTimer);
      resetCameraView = () => undefined;
      toggleProjection = () => undefined;
      canvas.removeEventListener('pointerup', handleFloorPick);
      viewCubeCanvas.removeEventListener('pointerdown', handleViewCubePointerDown);
      viewCubeCanvas.removeEventListener('pointermove', handleViewCubePointerMove);
      viewCubeCanvas.removeEventListener('pointerup', handleViewCubePointerUp);
      viewCubeCanvas.removeEventListener('pointercancel', handleViewCubePointerUp);
      viewCubeCanvas.removeEventListener('pointerleave', handleViewCubePointerLeave);
      resizeObserver.disconnect();
      controls.removeEventListener('change', handleControlsChange);
      controls.removeEventListener('start', handleControlsStart);
      controls.removeEventListener('end', handleControlsEnd);
      controls.dispose();
      transformControls.removeEventListener('objectChange', handleGizmoObjectChange);
      transformControls.removeEventListener('dragging-changed', handleGizmoDragging);
      transformControls.removeEventListener('mouseUp', commitTransform);
      transformControls.detach();
      transformControls.dispose();
      scene.remove(transformHelper);
      scene.remove(clipBox);
      clearSplat();
      clearMesh();
      sparkRenderer?.dispose();
      pointGeometry.dispose();
      pointMaterial.dispose();
      surfaceMaterial.dispose();
      shadedMaterial.dispose();
      wireMaterial.dispose();
      clipBoxGeometry.dispose();
      clipBoxMaterial.dispose();
      clipEdgeGeometry.dispose();
      clipEdgeMaterial.dispose();
      viewCubeGeometry.dispose();
      viewCubeEdgeGeometry.dispose();
      viewCubeEdgeMaterial.dispose();
      viewCubeTextures.forEach((texture) => texture.dispose());
      viewCubeMaterials.forEach((material) => material.dispose());
      viewCubeRenderer.dispose();
      renderer.dispose();
    };
  });
</script>

<div class:processing class:point-pick={floorPickMode} class:clip-edit={clipEditMode} class="viewer">
  <canvas class="scene-canvas" bind:this={canvas} aria-label="Interactive 3D reconstruction"></canvas>
  <div class:live class="viewer-hud top-left">
    <span class="pulse"></span>
    {processing ? renderMode === 'splat' ? `Training ${splatRepresentation === '3d' ? 'photoreal 3D' : 'surface 2D'} Gaussian splats` : 'Reconstructing geometry' : live ? liveLabel : renderMode === 'splat' ? splatError ? 'Splat preview failed' : splatReady ? `${splatRepresentation === '3d' ? '3D' : '2D'} Gaussian splat` : 'Loading splat' : renderMode === 'mesh' && mesh ? meshViewMode === 'wireframe' ? 'Wireframe' : meshViewMode === 'shaded' ? 'Shaded mesh' : meshViewMode === 'surface-wireframe' ? 'Mesh + wireframe' : 'Textured mesh' : points.length || packedFrame?.pointCount ? 'Point cloud' : 'Awaiting RGB-D frames'}
  </div>

  <div class="view-cube-shell">
    <canvas
      class="view-cube-canvas"
      bind:this={viewCubeCanvas}
      aria-label="View cube. Drag to orbit or click a face, edge, or corner to align the camera."
      title="Drag to orbit · Click a face, edge, or corner to snap"
    ></canvas>
    <button
      class:orthographic={projectionMode === 'orthographic'}
      class="projection-toggle"
      type="button"
      on:click={toggleProjection}
      aria-label={`Switch to ${projectionMode === 'orthographic' ? 'perspective' : 'orthographic'} projection`}
      title={`Switch to ${projectionMode === 'orthographic' ? 'perspective' : 'orthographic'} projection`}
    >{projectionMode === 'orthographic' ? 'ISO' : 'PERSP'}</button>
    <button class="view-home" type="button" on:click={resetCameraView} aria-label="Reset to isometric view" title="Reset to isometric view">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3.5 11.1 12 4l8.5 7.1M5.8 9.4v9.1h4.1v-5.2h4.2v5.2h4.1V9.4" />
      </svg>
    </button>
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
    <div class="viewer-hud bottom-left">Drag to orbit · Scroll to zoom</div>
  {:else}
    <div class="empty-state">
      <strong>{processing ? 'Preparing reconstruction geometry…' : renderMode === 'splat' ? splatError || 'No Gaussian splat yet' : renderMode === 'mesh' ? 'No reconstructed mesh yet' : live ? 'No valid depth in camera range' : 'No live depth points yet'}</strong>
      <span>{processing ? 'The viewer updates whenever the worker publishes a quality-gated snapshot.' : live ? emptyDetail || 'Aim the camera at a surface between its minimum range and the configured depth limit.' : 'Start capture or build the selected output.'}</span>
    </div>
  {/if}
  {#if processing}<div class="processing-scan"></div>{/if}
</div>

<style>
  .viewer { position: relative; width: 100%; height: 100%; min-height: 0; overflow: hidden; border: 1px solid #292e35; border-radius: 6px; background: #0b0d10; }
  .scene-canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; cursor: grab; }
  .scene-canvas:active { cursor: grabbing; }
  .point-pick .scene-canvas, .point-pick .scene-canvas:active { cursor: crosshair; }
  .clip-edit .scene-canvas { cursor: default; }
  .view-cube-shell { position: absolute; z-index: 7; top: 14px; right: 14px; width: 118px; height: 118px; pointer-events: none; }
  .view-cube-canvas { position: relative; display: block; width: 118px; height: 118px; cursor: grab; pointer-events: auto; touch-action: none; }
  .view-cube-canvas:active { cursor: grabbing; }
  .projection-toggle { position: absolute; z-index: 1; top: 1px; left: 1px; min-width: 37px; height: 27px; padding: 0 7px; border: 1px solid #343a43; border-radius: 4px; background: #171b20; color: #8e98a5; font: 800 7px/1 Inter, Segoe UI, sans-serif; letter-spacing: 0.06em; cursor: pointer; pointer-events: auto; }
  .projection-toggle.orthographic { border-color: #416c59; color: #67c49f; }
  .projection-toggle:hover { border-color: #52698f; background: #202630; color: #d2d8e1; }
  .projection-toggle:focus-visible { outline: 2px solid rgba(99, 199, 231, 0.75); outline-offset: 2px; }
  .view-home { position: absolute; z-index: 1; top: 1px; right: 1px; display: grid; place-items: center; width: 27px; height: 27px; padding: 0; border: 1px solid #343a43; border-radius: 4px; background: #171b20; color: #79a5f5; cursor: pointer; pointer-events: auto; }
  .view-home svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
  .view-home:hover { border-color: #52698f; background: #202630; color: #d2d8e1; }
  .view-home:focus-visible { outline: 2px solid rgba(99, 199, 231, 0.75); outline-offset: 2px; }
  .viewer-hud { position: absolute; display: flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid #30353d; border-radius: 4px; background: #171b20; color: #b5bbc4; font-size: 11px; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase; pointer-events: none; }
  .top-left { top: 16px; left: 16px; }
  .bottom-left { left: 16px; bottom: 16px; text-transform: none; letter-spacing: 0; }
  .floor-hint { left: 50%; bottom: 16px; transform: translateX(-50%); border-color: rgba(240, 183, 107, 0.35); color: #f0c68f; }
  .pulse { width: 7px; height: 7px; border-radius: 50%; background: #54b78d; }
  .processing .pulse { background: #f0b76b; animation: pulse 1.1s infinite; }
  .viewer-hud.live .pulse { animation: pulse 1.1s infinite; }
  .empty-state { position: absolute; inset: 0; display: grid; place-content: center; gap: 8px; color: #6c8593; text-align: center; pointer-events: none; }
  .empty-state strong { color: #a9bec8; font-size: 15px; }
  .empty-state span { font-size: 11px; }
  .asset-loading { position: absolute; z-index: 5; left: 50%; top: 50%; width: min(320px, calc(100% - 40px)); padding: 22px 24px; display: grid; justify-items: center; gap: 9px; transform: translate(-50%, -50%); border: 1px solid #343a43; border-radius: 6px; background: #171b20; color: #c4c9d0; text-align: center; pointer-events: none; }
  .asset-loading strong { color: #d4e7ef; font-size: 14px; }
  .asset-loading span { color: #78909d; font-size: 10px; }
  .asset-loading-spinner { width: 25px; height: 25px; border: 2px solid rgba(104, 195, 227, 0.2); border-top-color: #68c3e3; border-radius: 50%; animation: loading-spin 0.8s linear infinite; }
  .asset-loading-track { width: 100%; height: 3px; margin-top: 4px; overflow: hidden; border-radius: 3px; background: rgba(104, 195, 227, 0.12); }
  .asset-loading-track i { display: block; width: 38%; height: 100%; border-radius: inherit; background: #6c9eff; animation: loading-track 1.15s ease-in-out infinite; }
  .asset-loading-track.determinate i { background: #68c3e3; animation: none; transition: width 120ms linear; }
  .processing-scan { display: none; }
  @keyframes scan { to { transform: translateY(100%); } }
  @keyframes pulse { 50% { opacity: 0.4; transform: scale(0.8); } }
  @keyframes loading-spin { to { transform: rotate(360deg); } }
  @keyframes loading-track { from { transform: translateX(-110%); } to { transform: translateX(270%); } }
</style>
