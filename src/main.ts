import { mat4, vec3 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import basicWGSL from './shaders/basic.wgsl?raw'; // Raw String Import but only specific to Vite.
import { ArcballCamera, WASDCamera } from './camera';
import { HeadTrackedCamera } from './headTrackedCamera'; // Added head-tracked camera
import { createInputHandler } from './input';
import { loadAndProcessGLB } from './loadParseGLB';
import { RenderTarget } from './RenderTarget';
import { PostProcessEffect } from './postprocessing/PostProcessEffect';
import { PassThroughEffect } from './postprocessing/PassThroughEffect';
import { GrayscaleEffect } from './postprocessing/GrayscaleEffect';
import { FXAAEffect } from './postprocessing/FXAAEffect';
// Glow FX imports
import { BrightPassEffect } from './postprocessing/BrightPassEffect';
import { BlurEffect } from './postprocessing/GaussianBlurEffect';
import { GlowAddEffect } from './postprocessing/GlowAddEffect';
import { UnrealGlowEffect } from './postprocessing/UnrealGlowEffect';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

// const MESH_PATH = '/assets/meshes/light_color.glb';
const MESH_PATH = '/assets/meshes/cube_color.glb';

export class WebGPUApp{
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private presentationFormat!: GPUTextureFormat;
  private uniformBindGroup!: GPUBindGroup;
  private renderPassDescriptor!: GPURenderPassDescriptor;
  private cubeTexture!: GPUTexture;
  private cameras: { [key: string]: any };
  private aspect!: number;
  private params: { 
    type: 'arcball' | 'WASD' | 'head'; // added 'head'
    uTestValue: number; 
    uTestValue_02: number; 
    uGlow_Threshold: number;
    uGlow_ThresholdKnee: number; // Added for soft-knee threshold
    uGlow_Radius: number;
    uGlow_Intensity: number;
  } = {
    type: 'head',
    uTestValue: 1.0,
    uTestValue_02: 5.0,
    uGlow_Threshold: 0.5,
    uGlow_ThresholdKnee: 0.1,
    uGlow_Radius: 3.0,
    uGlow_Intensity: 0.5,
  };
  private uTime: number = 0.0;
  private gui: GUI;
  private lastFrameMS: number;
  private demoVerticesBuffer!: GPUBuffer;
  private loadVerticesBuffer!: GPUBuffer;
  private loadIndexBuffer!: GPUBuffer | undefined;
  private loadIndexCount!: number;
  private uniformBuffer!: GPUBuffer;
  private sceneUniformBuffer!: GPUBuffer;
  private objectUniformBuffer!: GPUBuffer;
  private viewMatrixBuffer!: GPUBuffer;
  private projectionMatrixBuffer!: GPUBuffer;
  private canvasSizeBuffer!: GPUBuffer;
  private uTimeBuffer!: GPUBuffer;
  private modelMatrixBuffer!: GPUBuffer;
  private uTestValueBuffer!: GPUBuffer;
  private uTestValue_02Buffer!: GPUBuffer;
  private loadVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[]; };
  private modelMatrix: Float32Array;
  private viewMatrix: Float32Array;
  private projectionMatrix: Float32Array;
  private depthTexture!: GPUTexture;
  private sampler!: GPUSampler;
  private newCameraType!: string;
  private oldCameraType!: string;
  private renderTarget_ping!: RenderTarget;
  private renderTarget_pong!: RenderTarget;
  private postProcessEffects: PostProcessEffect[] = [];
  private inputHandler!: () => { 
    digital: { forward: boolean, backward: boolean, left: boolean, right: boolean, up: boolean, down: boolean, };
    analog: { x: number; y: number; zoom: number; touching: boolean };
  };
  private static readonly CLEAR_COLOR = [0.1, 0.1, 0.1, 1.0];
  private static readonly CAMERA_POSITION = vec3.create(3, 2, 5);
  private passThroughEffect!: PassThroughEffect;
  // Glow FX Variables
  private brightPassEffect!: BrightPassEffect;
  private blurEffectH!: BlurEffect;
  private blurEffectV!: BlurEffect;
  private glowAddEffect!: GlowAddEffect;
  private unrealGlowEffect!: UnrealGlowEffect;
  private enableGlow: boolean = true; // or control with GUI
  // Head camera placeholder state (until real face tracking integration)
  private headYaw = 0;
  private headPitch = 0;
  private headDistance = 6;
  private headSettings = {
    yawLimit: 0.6,
    pitchLimit: 0.4,
    minDist: 2.0,
    maxDist: 15.0,
    invertYaw: true, // new
  };
  // Tracking extras
  private baselineIOD: number | null = null;
  private calibrationDistance: number = 6;
  private faceDetected = false;
  private lastFaceTime = 0;
  private faceLostGraceMS = 500;
  private webcam = document.getElementById("webcam") as HTMLVideoElement;
  private landmarkCanvas = document.getElementById("landmark-canvas") as HTMLCanvasElement;
  private landmarkCtx = this.landmarkCanvas.getContext("2d");
  private lastVideoTime: number = -1;
  private webcamRunning: boolean = false;
  private faceLandmarker?: FaceLandmarker;
  private faceLandmarkerLoaded: boolean = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.gui = new GUI();
    this.cameras = {
      arcball: new ArcballCamera({ position: WebGPUApp.CAMERA_POSITION }),
      WASD: new WASDCamera({ position: WebGPUApp.CAMERA_POSITION }),
      head: new HeadTrackedCamera({ distance: 6 }), // new camera
    };
    this.oldCameraType = this.params.type;
    this.lastFrameMS = Date.now();
    this.sampler = {} as GPUSampler;

     // The input handler
    this.inputHandler = createInputHandler(window, this.canvas);

    // Initialize matrices
    this.modelMatrix = mat4.identity();
    this.viewMatrix = mat4.identity();
    this.projectionMatrix = mat4.identity();

    this.webcam.addEventListener("loadeddata", () => {
      this.landmarkCanvas.width = this.webcam.videoWidth;
      this.landmarkCanvas.height = this.webcam.videoHeight;
    });

    this.setupAndRender();
  }

  public async setupAndRender() {
    await this.initializeWebGPU();
    this.initRenderTargetsForPP();
    await this.initLoadAndProcessGLB();
    this.initUniformBuffer();
    await this.loadTexture();
    this.initCam();
    this.initPipelineBindGrp();
    this.initializeGUI();
    this.setupEventListeners();
    this.renderFrame();
    this.enableCam();
  }



  private async predictWebcam() {
    if (!this.faceLandmarkerLoaded || !this.webcam || this.webcam.readyState !== 4) {
      requestAnimationFrame(this.predictWebcam.bind(this));
      return;
    }

    // Resize overlay if needed
    if (
      this.landmarkCanvas.width !== this.webcam.videoWidth ||
      this.landmarkCanvas.height !== this.webcam.videoHeight
    ) {
      this.landmarkCanvas.width = this.webcam.videoWidth;
      this.landmarkCanvas.height = this.webcam.videoHeight;
    }

    // Only run detection if the video frame has changed
    if (this.lastVideoTime !== this.webcam.currentTime) {
      this.lastVideoTime = this.webcam.currentTime;
      const nowInMs = performance.now();
      const results = await this.faceLandmarker!.detectForVideo(this.webcam, nowInMs);

      // Draw landmarks if detected
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const ctx = this.landmarkCtx;
        ctx!.clearRect(0, 0, this.landmarkCanvas.width, this.landmarkCanvas.height);
        ctx!.fillStyle = "red";
        for (const lm of results.faceLandmarks[0]) {
          const x = lm.x * this.landmarkCanvas.width;
          const y = lm.y * this.landmarkCanvas.height;
          ctx!.beginPath();
          ctx!.arc(x, y, 2, 0, 2 * Math.PI);
          ctx!.fill();
        }


        // // DEBUG: Emit particles directly from the 468 landmark points
        const landmarks = results.faceLandmarks[0];
        // const numLandmarks = landmarks.length;
        // const positions = new Float32Array(PARTICLE_COUNT * 4);
        // // For PARTICLE_COUNT > 468, repeat the landmark points
        // for (let i = 0; i < PARTICLE_COUNT; i++) {
        //   const lm = landmarks[i % numLandmarks];
        //   // Map x/y from [0,1] to [-1,1] (NDC), z as-is
        //   positions[i * 4 + 0] = lm.x * 2 - 1;
        //   positions[i * 4 + 1] = -(lm.y * 2 - 1); // flip y for NDC
        //   positions[i * 4 + 2] = lm.z ?? 0;
        //   positions[i * 4 + 3] = 1.0;
        // }

        // Update head camera pose
        if (this.params.type === 'head') {
          this.updateHeadPoseFromLandmarks(landmarks as any);
        }

      } else {
        // Clear overlay if no face
        this.landmarkCtx!.clearRect(0, 0, this.landmarkCanvas.width, this.landmarkCanvas.height);
      }
    }
    if (this.webcamRunning) requestAnimationFrame(this.predictWebcam.bind(this));
  }

  private enableCam() {
    const TARGET_WIDTH = 480; // pick 320, 480, or 640
    // Request webcam access and stream to the video element
    navigator.mediaDevices.getUserMedia({ 
      video: {
        width: { ideal: TARGET_WIDTH, max: 640 },
        height: { ideal: 360 }, // or leave out and let aspect follow camera
        facingMode: 'user',     // front camera on mobile
        frameRate: { ideal: 30, max: 60 },
      },
      audio: false,
    }).then((stream) => {
      this.webcam.srcObject = stream;
      this.webcam.addEventListener("loadeddata", async () => {
        // Match overlay size to intrinsic video
        if (this.landmarkCanvas) {
          this.landmarkCanvas.width = this.webcam.videoWidth;
          this.landmarkCanvas.height = this.webcam.videoHeight;
        }
        await this.loadFaceLandmarker();
        this.webcamRunning = true;
        this.predictWebcam();
      }, { once: true });
    });
  }

  private async loadFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      import.meta.env.BASE_URL + 'assets/wasm'
    );
    this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: import.meta.env.BASE_URL + 'assets/face_landmarker.task' },
      runningMode: 'VIDEO',
      numFaces: 1
    });
    this.faceLandmarkerLoaded = true;
  }

  private async initLoadAndProcessGLB() {
    const { interleavedData, indices, indexCount, vertexLayout } = await loadAndProcessGLB(MESH_PATH);
  
    // Create vertex buffer
    const vertexBuffer = this.device.createBuffer({
      size: interleavedData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(interleavedData);
    vertexBuffer.unmap();

    // Create index buffer if indices exist
    let indexBuffer: GPUBuffer | undefined = undefined;
    if (indices) {
      // Create index buffer
      // Pad index buffer size to next multiple of 4 for avoiding alignment issues
      // WebGPU requires buffer sizes to be a multiple of 4 bytes
      const paddedIndexBufferSize = Math.ceil(indices.byteLength / 4) * 4;

      indexBuffer = this.device.createBuffer({
        size: paddedIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint16Array(indexBuffer.getMappedRange()).set(indices);
      indexBuffer.unmap();
    }

    this.loadVerticesBuffer = vertexBuffer;
    this.loadIndexBuffer = indexBuffer;
    this.loadIndexCount = indexCount;
    this.loadVertexLayout = vertexLayout;
  }

  private initCam(){
    this.aspect = this.canvas.width / this.canvas.height;
    // Use GUI-controlled uTestValue_02 to derive vertical FOV
    this.updateProjectionFromParam();

    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);
  }

  private async loadTexture() {
    const response = await fetch('../assets/img/uv1.png');
    const imageBitmap = await createImageBitmap(await response.blob());

    this.cubeTexture = this.device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture: this.cubeTexture },
      [imageBitmap.width, imageBitmap.height]
    );
  }

  private initUniformBuffer() {
    // View Matrix
    this.viewMatrixBuffer = this.device.createBuffer({
      size: 16 * 4, // mat4x4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.viewMatrixBuffer, 0, this.viewMatrix.buffer);

    // Projection Matrix
    this.projectionMatrixBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);

    // Canvas Size
    this.canvasSizeBuffer = this.device.createBuffer({
      size: 2 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const canvasSize = new Float32Array([this.canvas.width, this.canvas.height]);
    this.device.queue.writeBuffer(this.canvasSizeBuffer, 0, canvasSize.buffer);

    // uTime
    this.uTimeBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTimeArr = new Float32Array([this.uTime]);
    this.device.queue.writeBuffer(this.uTimeBuffer, 0, uTimeArr.buffer);

    // Model Matrix
    this.modelMatrixBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.modelMatrixBuffer, 0, this.modelMatrix.buffer);

    // uTestValue
    this.uTestValueBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTestValueArr = new Float32Array([this.params.uTestValue]);
    this.device.queue.writeBuffer(this.uTestValueBuffer, 0, uTestValueArr.buffer);

    // uTestValue_02
    this.uTestValue_02Buffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uTestValue_02Arr = new Float32Array([this.params.uTestValue_02]);
    this.device.queue.writeBuffer(this.uTestValue_02Buffer, 0, uTestValue_02Arr.buffer);
  }

  private setupEventListeners() {
    window.addEventListener('resize', this.resize.bind(this));
  }

  private resize() {
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

  this.aspect = this.canvas.width / this.canvas.height;
  this.updateProjectionFromParam();
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
    });

    this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);

    const canvasSizeArray = new Float32Array([this.canvas.width, this.canvas.height]);
    this.device.queue.writeBuffer(this.canvasSizeBuffer, 0, canvasSizeArray.buffer);

    // Recreate the depth texture to match the new canvas size
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Resize the render targets
    this.renderTarget_ping.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);
    this.renderTarget_pong.resize(this.device, this.canvas.width, this.canvas.height, this.presentationFormat);

  }

  private initializeGUI() {
    this.gui.add(this.params, 'type', ['arcball', 'WASD', 'head']).onChange(() => { // added head option
      this.newCameraType = this.params.type;
      this.cameras[this.newCameraType].matrix = this.cameras[this.oldCameraType].matrix;
      this.oldCameraType = this.newCameraType
    });
    // Head camera folder (stub controls)
    const headFolder = this.gui.addFolder('Head Camera (stub)');
    headFolder.add(this.headSettings, 'yawLimit', 0.1, 1.2).step(0.01).name('YawLimit');
    headFolder.add(this.headSettings, 'pitchLimit', 0.1, 1.0).step(0.01).name('PitchLimit');
    headFolder.add(this.headSettings, 'minDist', 0.5, 5.0).step(0.1).name('MinDist');
    headFolder.add(this.headSettings, 'maxDist', 5.0, 25.0).step(0.1).name('MaxDist');
    headFolder.add(this.headSettings, 'invertYaw').name('InvertYaw');
    headFolder.add({ Calibrate: () => this.calibrateHead() }, 'Calibrate');
    headFolder.close();
      
    this.gui.add(this.params, 'uTestValue', 0.0, 1.0).step(0.01).onChange((value) => {
      this.updateFloatUniform( 'uTestValue', value );
    });
    this.gui.add(this.params, 'uTestValue_02', 3.0, 30.0).step(0.01).onChange((value) => {
      // Update GPU-side scalar if it's referenced separately
      this.updateFloatUniform('uTestValue_02', value);
      // Recompute the projection matrix with new parameter
      this.updateProjectionFromParam();
    });
    
    const glowFolder = this.gui.addFolder('Glow FX');
    glowFolder.add(this.params, 'uGlow_Threshold', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_ThresholdKnee', 0.0, 1.0).step(0.01).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Radius', 0.1, 20.0).step(0.1).onChange(() => this.updateGlowUniforms());
    glowFolder.add(this.params, 'uGlow_Intensity', 0.0, 1.0).step(0.001).onChange(() => this.updateGlowUniforms());
    glowFolder.open();
  }

  // Map uTestValue_02 -> vertical FOV and upload projection matrix
  private updateProjectionFromParam() {
    if (!this.aspect) this.aspect = this.canvas.width / this.canvas.height;
    // Prevent division by zero
    const p = Math.max(0.001, this.params.uTestValue_02);
    // Chosen relation: fovY = 2π / p. For p=5 -> ~72° ; p larger => smaller FOV (telephoto)
    let fovY = (2 * Math.PI) / p;
    // Clamp to practical range (5° .. 140°)
    const minFov = 5 * Math.PI / 180;
    const maxFov = 140 * Math.PI / 180;
    fovY = Math.min(Math.max(fovY, minFov), maxFov);
    this.projectionMatrix = mat4.perspective(fovY, this.aspect, 1, 100.0);
    if (this.projectionMatrixBuffer) {
      this.device.queue.writeBuffer(this.projectionMatrixBuffer, 0, this.projectionMatrix.buffer);
    }
  }

  private updateGlowUniforms() {
    this.brightPassEffect.setThreshold(this.params.uGlow_Threshold);
    this.brightPassEffect.setKnee(this.params.uGlow_ThresholdKnee);
    this.blurEffectH.setRadius(this.params.uGlow_Radius);
    this.blurEffectV.setRadius(this.params.uGlow_Radius);
    this.glowAddEffect.setIntensity(this.params.uGlow_Intensity);
  }

  private updateFloatUniform(key: keyof typeof this.params, value: number) {
    const updatedFloatArray = new Float32Array([value]);
    switch (key) {
      case 'uTestValue':
        this.device.queue.writeBuffer(this.uTestValueBuffer, 0, updatedFloatArray.buffer);
        break;
      case 'uTestValue_02':
        this.device.queue.writeBuffer(this.uTestValue_02Buffer, 0, updatedFloatArray.buffer);
        break;
      // Add more cases for other uniforms as needed
      default:
        console.error(`Unknown key: ${key}`);
        return;
    }
  }

  private calibrateHead() {
    this.baselineIOD = null;          // capture next frame
    this.calibrationDistance = this.headDistance;
  }

  private updateHeadPoseFromLandmarks(landmarks: { x: number; y: number; z?: number }[]) {
    if (landmarks.length < 264) return;
    const left = landmarks[33];
    const right = landmarks[263];
    const centerX = (left.x + right.x) * 0.5;
    const centerY = (left.y + right.y) * 0.5;
    const iod = Math.hypot(right.x - left.x, right.y - left.y);

    if (this.baselineIOD === null) {
      this.baselineIOD = iod;
    }

    const normX = (centerX - 0.5) * 2;
    const normY = (centerY - 0.5) * 2;
    let yaw = normX * this.headSettings.yawLimit;
    if (this.headSettings.invertYaw) yaw = -yaw;
    let pitch = normY * this.headSettings.pitchLimit;

    let distance = this.headDistance;
    if (this.baselineIOD && iod > 0.00001) {
      const scale = this.baselineIOD / iod;  // >1 means farther
      distance = this.calibrationDistance * scale;
      distance = Math.min(Math.max(distance, this.headSettings.minDist), this.headSettings.maxDist);
    }

    this.headYaw = Math.min(Math.max(yaw, -this.headSettings.yawLimit), this.headSettings.yawLimit);
    this.headPitch = Math.min(Math.max(pitch, -this.headSettings.pitchLimit), this.headSettings.pitchLimit);
    this.headDistance = distance;

    (this.cameras['head'] as HeadTrackedCamera).setPose({
      yaw: this.headYaw,
      pitch: this.headPitch,
      distance: this.headDistance
    });
  }

  private async initializeWebGPU() {
    const adapter = await navigator.gpu?.requestAdapter({ featureLevel: 'compatibility' });
    this.device = await adapter?.requestDevice() as GPUDevice;

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const devicePixelRatio = window.devicePixelRatio;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    });

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      // Wrap UVs instead of clamping (prevents edge pixel stretch when >1 or <0)
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
    });

    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: undefined, // Assigned later
          clearValue: WebGPUApp.CLEAR_COLOR,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ] as Iterable< GPURenderPassColorAttachment | null | undefined>,
      depthStencilAttachment: {
        view: this.depthTexture.createView(), // Assign a valid GPUTextureView
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };
  }

  private initPipelineBindGrp() {

    const uniformBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // viewMatrix
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // projectionMatrix
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // canvasSize
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTime
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // modelMatrix
        { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTestValue
        { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, // uTestValue_02
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }, // Sampler
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // Texture
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [uniformBindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: basicWGSL }),
        entryPoint: 'vertex_main',
        buffers: [{
          arrayStride: this.loadVertexLayout.arrayStride,
          attributes: this.loadVertexLayout.attributes,
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: basicWGSL }),
        entryPoint: 'fragment_main',
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.uniformBindGroup = this.device.createBindGroup({
      layout: uniformBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.viewMatrixBuffer } },
        { binding: 1, resource: { buffer: this.projectionMatrixBuffer } },
        { binding: 2, resource: { buffer: this.canvasSizeBuffer } },
        { binding: 3, resource: { buffer: this.uTimeBuffer } },
        { binding: 4, resource: { buffer: this.modelMatrixBuffer } },
        { binding: 5, resource: { buffer: this.uTestValueBuffer } },
        { binding: 6, resource: { buffer: this.uTestValue_02Buffer } },
        { binding: 7, resource: this.sampler },
        { binding: 8, resource: this.cubeTexture.createView() },
      ],
    });
  }

  private getViewMatrix(deltaTime: number) {
    const camera = this.cameras[this.params.type];
    const input = this.inputHandler();
    if (this.params.type === 'head') {
      const headCam = camera as HeadTrackedCamera;
      if (!this.faceDetected) {
        // Mouse fallback
        this.headYaw += input.analog.x * 0.002;
        this.headPitch += input.analog.y * 0.002;
        this.headYaw = Math.min(Math.max(this.headYaw, -this.headSettings.yawLimit), this.headSettings.yawLimit);
        this.headPitch = Math.min(Math.max(this.headPitch, -this.headSettings.pitchLimit), this.headSettings.pitchLimit);
        if (input.analog.zoom !== 0) {
          this.headDistance *= 1 + input.analog.zoom * 0.05;
          this.headDistance = Math.min(Math.max(this.headDistance, this.headSettings.minDist), this.headSettings.maxDist);
        }
        headCam.setPose({ yaw: this.headYaw, pitch: this.headPitch, distance: this.headDistance });
      }
    }
    return camera.update(deltaTime, input);
  }

  private initRenderTargetsForPP() {
    // Create ping-pong render targets
    this.renderTarget_ping = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );
    this.renderTarget_pong = new RenderTarget(
      this.device,
      this.canvas.width,
      this.canvas.height,
      this.presentationFormat
    );

    // Init useful pass-through effect 
    this.passThroughEffect = new PassThroughEffect(this.device, this.presentationFormat, this.sampler);

    this.brightPassEffect = new BrightPassEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Threshold, this.params.uGlow_ThresholdKnee);
    // Add post-processing effects
    this.postProcessEffects.push(
      // new GrayscaleEffect(this.device, this.presentationFormat, this.sampler),
      this.brightPassEffect,
      new FXAAEffect(this.device, this.presentationFormat, this.sampler, [this.canvas.width, this.canvas.height]),
    );

    this.blurEffectH = new BlurEffect(this.device, this.presentationFormat, this.sampler, [1.0, 0.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.blurEffectV = new BlurEffect(this.device, this.presentationFormat, this.sampler, [0.0, 1.0], [1 / this.canvas.width, 1 / this.canvas.height], this.params.uGlow_Radius );
    this.glowAddEffect = new GlowAddEffect(this.device, this.presentationFormat, this.sampler, this.params.uGlow_Intensity );
    this.unrealGlowEffect = new UnrealGlowEffect(
      this.device,
      this.presentationFormat,
      this.sampler,
      this.canvas.width,
      this.canvas.height,
      4, // levels, adjust as needed
      this.brightPassEffect,
      this.blurEffectH,
      this.blurEffectV,
      this.glowAddEffect,
      this.passThroughEffect
    );
  }

  private renderFrame() {
    const now = Date.now();
    const deltaTime = (now - this.lastFrameMS) / 1000;
    this.lastFrameMS = now;

    // Update the uniform uTime value
    this.uTime += deltaTime;
    const uTimeFloatArray = new Float32Array([this.uTime]);
    this.device.queue.writeBuffer(this.uTimeBuffer, 0, new Float32Array([this.uTime]).buffer);

    this.viewMatrix = this.getViewMatrix(deltaTime);
    this.device.queue.writeBuffer(this.viewMatrixBuffer, 0, this.viewMatrix.buffer);
    // Set up a render pass target based on post-processing effects
    if (this.postProcessEffects.length === 0) {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context.getCurrentTexture().createView();
    } else {
      (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.renderTarget_ping.view;
    }

    // Update the depth attachment view
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthTexture.createView();

    const commandEncoder = this.device.createCommandEncoder();
    
    const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);

    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.uniformBindGroup);
    passEncoder.setVertexBuffer(0, this.loadVerticesBuffer);
    passEncoder.setIndexBuffer(this.loadIndexBuffer!, 'uint16');
    passEncoder.drawIndexed(this.loadIndexCount);
    passEncoder.end();

    // Apply post-processing effects if any
    let finalOutputView = this.renderTarget_ping.view;
    if (this.postProcessEffects.length > 0) {
      let inputView = this.renderTarget_ping.view;
      let outputView = this.renderTarget_pong.view;
      for (let i = 0; i < this.postProcessEffects.length; i++) {
        const isLast = i === this.postProcessEffects.length - 1;

        if(!this.enableGlow) { // Only use single output for PostProcessEffects
          finalOutputView = isLast ? this.context.getCurrentTexture().createView() : outputView;
        } else { // Make sure to continue using ping-pong buffers when applying glowFX afterwards
          finalOutputView = outputView;
        }
        
        this.postProcessEffects[i].apply(
          commandEncoder,
          { A: inputView },
          finalOutputView,
          [this.canvas.width, this.canvas.height]
        );
        if (!isLast) {
          [inputView, outputView] = [outputView, inputView];
        }
      }
      if (this.enableGlow) {
        this.unrealGlowEffect.apply(
          commandEncoder,
          finalOutputView,
          this.context.getCurrentTexture().createView()
        );
      }
    }

    this.device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(this.renderFrame.bind(this));
  }
}

const app = new WebGPUApp(document.getElementById('app') as HTMLCanvasElement);