<script lang="ts">
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import type { SparkRenderer as SparkRendererInstance, SplatMesh as SplatMeshInstance } from '@sparkjsdev/spark';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import type { MeshViewMode, PackedPreviewFrame, PreviewMesh, PreviewPoint } from '../types';

  type RenderMode = 'points' | 'mesh' | 'splat';
  type AssetLoadingState = 'points' | 'mesh' | 'mesh-texture' | 'splat' | 'splat-gpu' | null;

  export let points: PreviewPoint[] = [];
  export let packedFrame: PackedPreviewFrame | null = null;
  export let processing = false;
  export let live = false;
  export let pointSize = 0.034;
  export let opacity = 0.92;
  export let showColors = true;
  export let meshViewMode: MeshViewMode = 'surface';
  export let renderMode: RenderMode = 'points';
  export let mesh: PreviewMesh | null = null;
  export let splatBytes: Uint8Array | null = null;
  export let assetLoading: 'points' | 'mesh' | 'splat' | null = null;

  let canvas: HTMLCanvasElement;
  let setPoints: (next: PreviewPoint[]) => void = () => undefined;
  let setPackedPoints: (next: PackedPreviewFrame) => void = () => undefined;
  let setMaterial: (size: number, alpha: number, colors: boolean) => void = () => undefined;
  let setMesh: (next: PreviewMesh | null) => void = () => undefined;
  let setMeshMode: (next: MeshViewMode) => void = () => undefined;
  let setSplat: (next: Uint8Array | null) => void = () => undefined;
  let setRenderMode: (next: RenderMode) => void = () => undefined;
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
    setRenderMode = (next) => {
      pointCloud.visible = next === 'points';
      meshGroup.visible = next === 'mesh';
      splatGroup.visible = next === 'splat';
      grid.visible = next !== 'splat';
      applyPixelRatio(next === 'splat' ? qualityPixelRatio : Math.min(window.devicePixelRatio, 2));
      updateMeshAppearance();
      invalidate();
    };

    if (packedFrame) setPackedPoints(packedFrame);
    else setPoints(points);
    setMesh(mesh);
    setSplat(splatBytes);
    setMaterial(pointSize, opacity, showColors);
    setMeshMode(meshViewMode);
    setRenderMode(renderMode);

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
      resizeObserver.disconnect();
      controls.removeEventListener('change', invalidate);
      controls.removeEventListener('start', interactionStart);
      controls.removeEventListener('end', interactionEnd);
      controls.dispose();
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

<div class:processing class="viewer">
  <canvas bind:this={canvas} aria-label="Interactive 3D reconstruction"></canvas>
  <div class:live class="viewer-hud top-left">
    <span class="pulse"></span>
    {processing ? renderMode === 'splat' ? 'Training 2D Gaussian splats' : 'Reconstructing geometry' : live ? 'Live RGB-D reconstruction' : renderMode === 'splat' ? splatError ? 'Splat preview failed' : splatReady ? '2D Gaussian splat' : 'Loading splat' : renderMode === 'mesh' && mesh ? meshViewMode === 'wireframe' ? 'Wireframe' : meshViewMode === 'shaded' ? 'Shaded mesh' : meshViewMode === 'surface-wireframe' ? 'Mesh + wireframe' : 'Textured mesh' : points.length || packedFrame?.pointCount ? 'Point cloud' : 'Awaiting RGB-D frames'}
  </div>

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
      <strong>{processing ? 'Preparing reconstruction geometry…' : renderMode === 'splat' ? splatError || 'No Gaussian splat yet' : renderMode === 'mesh' ? 'No reconstructed mesh yet' : 'No live depth points yet'}</strong>
      <span>{processing ? 'The viewer updates whenever the worker publishes a quality-gated snapshot.' : 'Start capture or build the selected output.'}</span>
    </div>
  {/if}
  {#if processing}<div class="processing-scan"></div>{/if}
</div>

<style>
  .viewer { position: relative; width: 100%; height: 100%; min-height: 0; overflow: hidden; border-radius: 22px; background: #07111c; box-shadow: inset 0 0 0 1px rgba(139, 193, 216, 0.08); }
  canvas { position: absolute; inset: 0; display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .viewer-hud { position: absolute; display: flex; align-items: center; gap: 8px; padding: 8px 11px; border: 1px solid rgba(157, 204, 223, 0.12); border-radius: 10px; background: rgba(5, 15, 25, 0.68); color: #adc3d0; font-size: 11px; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase; backdrop-filter: blur(12px); pointer-events: none; }
  .top-left { top: 16px; left: 16px; }
  .bottom-right { right: 16px; bottom: 16px; text-transform: none; letter-spacing: 0; }
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
