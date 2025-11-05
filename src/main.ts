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
import { DepthCompEffect } from './postprocessing/DepthCompEffect';
// Glow FX imports
import { BrightPassEffect } from './postprocessing/BrightPassEffect';
import { BlurEffect } from './postprocessing/GaussianBlurEffect';
import { GlowAddEffect } from './postprocessing/GlowAddEffect';
import { UnrealGlowEffect } from './postprocessing/UnrealGlowEffect';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

// const MESH_PATH = '/assets/meshes/light_color.glb';
const MESH_PATH = '/assets/meshes/tesseract_color_v03.glb';
const CAM_MESH_PATH = '/assets/meshes/cam_1000.glb';

export class WebGPUApp{
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private pipeline!: GPURenderPipeline;
  private camPipeline!: GPURenderPipeline;
  private presentationFormat!: GPUTextureFormat;
  private uniformBindGroup!: GPUBindGroup;
  private camUniformBindGroup!: GPUBindGroup;
  private renderPassDescriptor!: GPURenderPassDescriptor;
  private cubeTexture!: GPUTexture;
  private maskTexture!: GPUTexture;
  private uniformBindGroupLayout!: GPUBindGroupLayout;
  private camUniformBindGroupLayout!: GPUBindGroupLayout;
  private videoTextureReady: boolean = false; 
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
  private camLoadVerticesBuffer!: GPUBuffer;
  private loadIndexBuffer!: GPUBuffer | undefined;
  private camLoadIndexBuffer!: GPUBuffer | undefined;
  private loadIndexCount!: number;
  private camLoadIndexCount!: number;
  private uniformBuffer!: GPUBuffer;
  private sceneUniformBuffer!: GPUBuffer;
  private objectUniformBuffer!: GPUBuffer;
  private viewMatrixBuffer!: GPUBuffer;
  private projectionMatrixBuffer!: GPUBuffer;
  private canvasSizeBuffer!: GPUBuffer;
  private uTimeBuffer!: GPUBuffer;
  private modelMatrixBuffer!: GPUBuffer;
  private camModelMatrixBuffer!: GPUBuffer;
  private uTestValueBuffer!: GPUBuffer;
  private uTestValue_02Buffer!: GPUBuffer;
  private loadVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[]; };
  private camLoadVertexLayout!: { arrayStride: number; attributes: GPUVertexAttribute[]; };
  private modelMatrix: Float32Array;
  private camModelMatrix: Float32Array;
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
  private depthCompEffect!: DepthCompEffect;
  // Glow FX Variables
  private brightPassEffect!: BrightPassEffect;
  private blurEffectH!: BlurEffect;
  private blurEffectV!: BlurEffect;
  private glowAddEffect!: GlowAddEffect;
  private unrealGlowEffect!: UnrealGlowEffect;
  private enableGlow: boolean = false; // or control with GUI
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
      head: new HeadTrackedCamera({ distance: 6, rotationHalfLife: 0.02, distanceHalfLife: 0.01 }), // new camera
    };
    this.oldCameraType = this.params.type;
    this.lastFrameMS = Date.now();
    this.sampler = {} as GPUSampler;

     // The input handler
    this.inputHandler = createInputHandler(window, this.canvas);

    // Initialize matrices
    this.modelMatrix = mat4.identity();
    this.camModelMatrix = mat4.identity();
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

    // // Resize overlay if needed
    // if (
    //   this.landmarkCanvas.width !== this.webcam.videoWidth ||
    //   this.landmarkCanvas.height !== this.webcam.videoHeight
    // ) {
    //   this.landmarkCanvas.width = this.webcam.videoWidth;
    //   this.landmarkCanvas.height = this.webcam.videoHeight;
    // }

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
        // if (this.params.type === 'head') {
        //   this.updateHeadPoseFromLandmarks(landmarks as any);
        // }
        this.updateHeadPoseFromLandmarks(landmarks as any);

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
      video: true,
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
        this.initVideoTextureFromWebcam();
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
      new Uint32Array(indexBuffer.getMappedRange()).set(indices);
      indexBuffer.unmap();
    }
    
    this.loadVerticesBuffer = vertexBuffer;
    this.loadIndexBuffer = indexBuffer;
    this.loadIndexCount = indexCount;
    this.loadVertexLayout = vertexLayout;

    const { interleavedData: camInterleavedData, indices: camIndices, indexCount: camIndexCount, vertexLayout: camVertexLayout } = await loadAndProcessGLB(CAM_MESH_PATH);

    const camVertexBuffer = this.device.createBuffer({
      size: camInterleavedData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(camVertexBuffer.getMappedRange()).set(camInterleavedData);
    camVertexBuffer.unmap();
    let camIndexBuffer: GPUBuffer | undefined = undefined;
    if (camIndices) {
      const paddedIndexBufferSize = Math.ceil(camIndices.byteLength / 4) * 4;

      camIndexBuffer = this.device.createBuffer({
        size: paddedIndexBufferSize,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint32Array(camIndexBuffer.getMappedRange()).set(camIndices);
      camIndexBuffer.unmap();
    }

    this.camLoadVerticesBuffer = camVertexBuffer;
    this.camLoadIndexBuffer = camIndexBuffer;
    this.camLoadIndexCount = camIndexCount;
    this.camLoadVertexLayout = camVertexLayout;
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

  private initVideoTextureFromWebcam() {
    const w = this.webcam.videoWidth | 0;
    const h = this.webcam.videoHeight | 0;
    if (!w || !h) return;

    this.cubeTexture = this.device.createTexture({
      size: [w, h, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Recreate the bind group with the new texture view
    this.uniformBindGroup = this.device.createBindGroup({
      layout: this.uniformBindGroupLayout,
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
        { binding: 9, resource: this.maskTexture.createView() },
      ],
    });

    this.videoTextureReady = true;
  }

  private async loadTexture() {
    const response_01 = await fetch('../assets/img/uv1.png');
    const imageBitmap_01 = await createImageBitmap(await response_01.blob());
    const response_02 = await fetch('../assets/img/noise_mask.png');
    const imageBitmap_02 = await createImageBitmap(await response_02.blob());

    this.cubeTexture = this.device.createTexture({
      size: [imageBitmap_01.width, imageBitmap_01.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.maskTexture = this.device.createTexture({
      size: [imageBitmap_02.width, imageBitmap_02.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap_01 },
      { texture: this.cubeTexture },
      [imageBitmap_01.width, imageBitmap_01.height]
    );
    this.device.queue.copyExternalImageToTexture(
      { source: imageBitmap_02 },
      { texture: this.maskTexture },
      [imageBitmap_02.width, imageBitmap_02.height]
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
    
    // Head Camera Model Matrix
    this.camModelMatrixBuffer = this.device.createBuffer({
      size: 16 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.camModelMatrixBuffer, 0, this.camModelMatrix.buffer);

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
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Update the depth attachment
    this.renderPassDescriptor.depthStencilAttachment!.view = this.depthTexture.createView();

    // Keep the effect’s depth view in sync
    if (this.depthCompEffect) {
      this.depthCompEffect.setDepthView(this.depthTexture.createView());
    }

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
    this.projectionMatrix = mat4.perspective(fovY, this.aspect, 1, 1000.0);
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
    const left = landmarks[33]; // outer edge of the right eye
    const right = landmarks[263];
    const centerX = (left.x + right.x) * 0.5; // X center between eyes
    const centerY = (left.y + right.y) * 0.5; // Y center between eyes
    const iod = Math.hypot(right.x - left.x, right.y - left.y);

    if (this.baselineIOD === null) {
      this.baselineIOD = iod;
    }

    // console.log(centerX)

    const normX = (centerX - 0.5) * 2;
    const normY = (centerY - 0.5) * 2;
    let yaw = normX * this.headSettings.yawLimit;
    if (this.headSettings.invertYaw) yaw = -yaw;
    let pitch = normY * this.headSettings.pitchLimit;

    let distance = this.headDistance;
    if (this.baselineIOD && iod > 0.00001) {
      const distanceGain = 3.0; // >1 amplifies forward/back effect
      const ratio = this.baselineIOD! / iod;       // >1 => farther
      const gamma = 1.6;                               // >1 amplifies; 0<gamma<1 softens
      const ratioNL = Math.pow(Math.min(Math.max(ratio, 0.01), 10.0), gamma); // clamp, then power
      distance = this.calibrationDistance * ratioNL * distanceGain;
    //   distance = Math.min(Math.max(distance, this.headSettings.minDist), this.headSettings.maxDist);
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
    // const adapter = await navigator.gpu?.requestAdapter({ featureLevel: 'compatibility' });
    const adapter = await navigator.gpu?.requestAdapter();
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
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
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

    this.uniformBindGroupLayout = this.device.createBindGroupLayout({
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
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // Texture
      ],
    });

    this.camUniformBindGroupLayout = this.device.createBindGroupLayout({
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
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // Texture
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.uniformBindGroupLayout],
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

    this.camPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.uniformBindGroupLayout],
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
      layout: this.uniformBindGroupLayout,
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
        { binding: 9, resource: this.maskTexture.createView() },
      ],
    });
    
    this.camUniformBindGroup = this.device.createBindGroup({
      layout: this.camUniformBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.viewMatrixBuffer } },
        { binding: 1, resource: { buffer: this.projectionMatrixBuffer } },
        { binding: 2, resource: { buffer: this.canvasSizeBuffer } },
        { binding: 3, resource: { buffer: this.uTimeBuffer } },
        { binding: 4, resource: { buffer: this.camModelMatrixBuffer } },
        { binding: 5, resource: { buffer: this.uTestValueBuffer } },
        { binding: 6, resource: { buffer: this.uTestValue_02Buffer } },
        { binding: 7, resource: this.sampler },
        { binding: 8, resource: this.cubeTexture.createView() },
        { binding: 9, resource: this.maskTexture.createView() },
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
    } else {
      // Even when viewing with Arcball/WASD, advance the head camera so the icon keeps moving
      (this.cameras['head'] as HeadTrackedCamera).update(deltaTime, input);
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
    this.depthCompEffect = new DepthCompEffect(this.device, this.presentationFormat, this.sampler, this.depthTexture.createView());
    // Add post-processing effects
    this.postProcessEffects.push(
      // new GrayscaleEffect(this.device, this.presentationFormat, this.sampler),
      // this.brightPassEffect,
      new FXAAEffect(this.device, this.presentationFormat, this.sampler, [this.canvas.width, this.canvas.height]),
      this.depthCompEffect,
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

  // Build and upload the model matrix for the head camera icon
  private writeHeadIconModelMatrix(scale = 0.1, offset = 0.0) {
    const headCam = this.cameras['head'] as HeadTrackedCamera;

    const camModel = mat4.create();
    mat4.copy(headCam.matrix, camModel);

    // Scale orientation columns (right, up, back). Leave position unscaled.
    camModel[0]  *= scale; camModel[1]  *= scale; camModel[2]  *= scale;   // right
    camModel[4]  *= scale; camModel[5]  *= scale; camModel[6]  *= scale;   // up
    camModel[8]  *= scale; camModel[9]  *= scale; camModel[10] *= scale;   // back

    // Optional: push the icon forward (toward camera's look direction) to ensure visibility
    if (offset !== 0) {
      // forward = -back
      camModel[12] += -camModel[8]  * offset;
      camModel[13] += -camModel[9]  * offset;
      camModel[14] += -camModel[10] * offset;
    }

    // Upload to the shared model matrix uniform
    this.device.queue.writeBuffer(this.camModelMatrixBuffer, 0, camModel.buffer);
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

    if (this.videoTextureReady && this.webcam.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      this.device.queue.copyExternalImageToTexture(
        { source: this.webcam, flipY: false }, // set flipY=false if your UVs already expect top-left origin
        { texture: this.cubeTexture },
        [this.webcam.videoWidth, this.webcam.videoHeight]
      );
    }
    
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
    passEncoder.setIndexBuffer(this.loadIndexBuffer!, 'uint32');
    passEncoder.drawIndexed(this.loadIndexCount);

    this.writeHeadIconModelMatrix(0.1, 0.0);
    passEncoder.setBindGroup(0, this.camUniformBindGroup);
    passEncoder.setVertexBuffer(0, this.camLoadVerticesBuffer);
    passEncoder.setIndexBuffer(this.camLoadIndexBuffer!, 'uint32');
    passEncoder.drawIndexed(this.camLoadIndexCount);
    passEncoder.end();

    // Provide depth to the effect (view can be recreated each frame)
    this.depthCompEffect.setDepthView(this.depthTexture.createView());

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