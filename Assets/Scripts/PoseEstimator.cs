using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;
using Unity.Barracuda;

public class PoseEstimator : MonoBehaviour
{
    public enum ModelType
    {
        MobileNet,
        ResNet50,
        LightMobileNet,
        OpenPoseSinglePoseOnly
    }

    public enum EstimationType
    {
        MultiPose,
        SinglePose
    }


    [Tooltip("The screen for viewing preprocessed images")]
    public Transform videoScreen;

    [Tooltip("The ComputeShader that will perform the model-specific preprocessing")]
    public ComputeShader posenetShader;

    [Tooltip("The model architecture used")]
    public ModelType modelType = ModelType.MobileNet;

    [Tooltip("Use GPU for preprocessing")]
    public bool useGPU = true;

    [Tooltip("The dimensions of the image being fed to the model")]
    public Vector2Int imageDims = new Vector2Int(256, 256);

    [Tooltip("The MobileNet model")]
    public NNModel mobileNetModelAsset;

    [Tooltip("The ResNet50 model")]
    public NNModel resnetModelAsset;

    [Tooltip("The LightMobileNet model")]
    public NNModel lightMobileNetModelAsset;

    [Tooltip("The OpenPose model (single-pose only)")]
    public NNModel openposeModelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The type of pose estimation to be performed")]
    public EstimationType estimationType = EstimationType.SinglePose;

    [Tooltip("The maximum number of posees to estimate")]
    [Range(1, 20)]
    public int maxPoses = 3;

    [Tooltip("The score threshold for multipose estimation")]
    [Range(0, 1.0f)]
    public float scoreThreshold = 0.25f;

    [Tooltip("Non-maximum suppression part distance")]
    public int nmsRadius = 100;

    [Tooltip("The size of the pose skeleton key points")]
    public float pointScale = 10f;

    [Tooltip("The width of the pose skeleton lines")]
    public float lineWidth = 5f;

    [Tooltip("The minimum confidence level required to display the key point")]
    [Range(0, 100)]
    public int minConfidence = 30;

    [Tooltip("Temporal regularization based on probabilistic estimation of movement")]
    public bool UseKalmanFiltering = false;

    [Tooltip("Parameter related to Kalman filter Q (temporal regularization)")]
    [Range(0.001f, 0.5f)]
    public float KalmanParamQ = 0.015f;

    [Tooltip("Parameter related to Kalman filter R (temporal regularization)")]
    [Range(0.001f, 0.5f)]
    public float KalmanParamR = 0.015f;

    // The dimensions of the current video source
    private Vector2Int videoDims;

    // The source video texture
    private RenderTexture videoTexture;

    // Target dimensions for model input
    private Vector2Int targetDims;

    // Used to scale the input image dimensions while maintaining aspect ratio
    private float aspectRatioScale;

    // The texture used to create input tensor
    private RenderTexture rTex;

    // The preprocessing function for the current model type
    private System.Action<float[]> preProcessFunction;

    // Stores the input data for the model
    private Tensor input;

    /// <summary>
    /// Keeps track of the current inference backend, model execution interface, 
    /// and model type
    /// </summary>
    private struct Engine
    {
        public WorkerFactory.Type workerType;
        public IWorker worker;
        public ModelType modelType;

        public Engine(WorkerFactory.Type workerType, Model model, ModelType modelType)
        {
            this.workerType = workerType;
            worker = WorkerFactory.CreateWorker(workerType, model);
            this.modelType = modelType;
        }
    }

    // The interface used to execute the neural network
    private Engine engine;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer;

    // The name for the offsets layer in the model asset
    private string offsetsLayer;

    // The name for the forwards displacement layer in the model asset
    private string displacementFWDLayer;

    // The name for the backwards displacement layer in the model asset
    private string displacementBWDLayer;

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "heatmap_predictions";

    // Stores the current estimated 2D keypoint locations in videoTexture
    private Utils.Keypoint[][] poses;
    private Utils.Keypoint[][] previous_poses;

    // Array of pose skeletons
    private PoseSkeleton[] skeletons;


    /// <summary>
    /// Prepares the videoScreen GameObject to display the chosen video source.
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    private void InitializeVideoScreen(int width, int height)
    {
        // Set the render mode for the video player
        videoScreen.GetComponent<VideoPlayer>().renderMode = VideoRenderMode.RenderTexture;

        // Use new videoTexture for Video Player
        videoScreen.GetComponent<VideoPlayer>().targetTexture = videoTexture;

        // Apply the new videoTexture to the VideoScreen Gameobject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.shader = Shader.Find("Unlit/Texture");
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);
        
        // Adjust the VideoScreen for the new videoTexture
        videoScreen.localScale = new Vector3(width, height, videoScreen.localScale.z);
        videoScreen.position = new Vector3(width / 2, height / 2, 1);
    }

    /// <summary>
    /// Resizes and positions the in-game Camera to accommodate the video dimensions
    /// </summary>
    private void InitializeCamera()
    {
        // Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(videoDims.x / 2, videoDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        mainCamera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoDims.y / 2;
    }


    /// <summary>
    /// Updates the output layer names based on the selected model architecture
    /// and initializes the Barracuda inference engine witht the selected model.
    /// </summary>
    private void StartEngine()
    {
        // The compiled model used for performing inference
        Model m_RunTimeModel;

        if ((modelType == ModelType.MobileNet) || (modelType == ModelType.MobileNet))
        {
            preProcessFunction = Utils.PreprocessMobileNet;
            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(mobileNetModelAsset);
            displacementFWDLayer = m_RunTimeModel.outputs[2];
            displacementBWDLayer = m_RunTimeModel.outputs[3];
        }
        else if (modelType == ModelType.OpenPoseSinglePoseOnly)
        {
            preProcessFunction = Utils.PreprocessOpenPose;
            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(openposeModelAsset);
            displacementFWDLayer = m_RunTimeModel.outputs[3];
            displacementBWDLayer = m_RunTimeModel.outputs[2];
        }
        else 
        {
            preProcessFunction = Utils.PreprocessResNet;
            // Compile the model asset into an object oriented representation
            m_RunTimeModel = ModelLoader.Load(resnetModelAsset);
            displacementFWDLayer = m_RunTimeModel.outputs[3];
            displacementBWDLayer = m_RunTimeModel.outputs[2];

        }

        heatmapLayer = m_RunTimeModel.outputs[0];
        offsetsLayer = m_RunTimeModel.outputs[1];

        // Create a model builder to modify the m_RunTimeModel
        // (could have modified the model itself instead. Maybe if I have time)
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);
        modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        // Validate if backend is supported, otherwise use fallback type.
        workerType = WorkerFactory.ValidateType(workerType);

        // Create a worker that will execute the model with the selected backend
        engine = new Engine(workerType, modelBuilder.model, modelType);
    }

    // Initialize pose skeletons
    private void DrawSkeletons()
    {
        // Initialize the list of pose skeletons
        if (estimationType == EstimationType.SinglePose) maxPoses = 1;
        skeletons = new PoseSkeleton[maxPoses];

        // Populate the list of pose skeletons
        for (int i = 0; i < maxPoses; i++) skeletons[i] = new PoseSkeleton(pointScale, lineWidth);
    }


    // Start is called before the first frame update
    void Start()
    {

        // Update the videoDims.y
        videoDims.x = (int)videoScreen.GetComponent<VideoPlayer>().width;
        videoDims.y = (int)videoScreen.GetComponent<VideoPlayer>().height;

        // Create a new videoTexture using the current video dimensions
        videoTexture = RenderTexture.GetTemporary(videoDims.x, videoDims.y, 24, RenderTextureFormat.ARGBHalf);

        // Initialize the videoScreen
        InitializeVideoScreen(videoDims.x, videoDims.y);

        // Adjust the camera based on the source video dimensions
        InitializeCamera();

        // Adjust the input dimensions to maintain the source aspect ratio
        aspectRatioScale = (float)videoTexture.width / videoTexture.height;
        targetDims.x = (int)(imageDims.y * aspectRatioScale);
        imageDims.x = targetDims.x;

        // Initialize the RenderTexture that will store the processed input image
        rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);

        // Initialize the Barracuda inference engine based on the selected model
        StartEngine();
        DrawSkeletons();
    }


    // Process the provided image using the specified function on the GPU
    // (from tutorial)
    private void ProcessImageGPU(RenderTexture image, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }

    // Calls the appropriate preprocessing function to prepare 
    // -> (this come from a tutorial)
    private void ProcessImage(RenderTexture image)
    {
        if (useGPU)
        {
            // Apply preprocessing steps
            ProcessImageGPU(image, preProcessFunction.Method.Name);
            input = new Tensor(image, 3); //if you don't put 3 it crashes on my mac. Known issue.
        }
        else
        {
            // Create a Tensor of shape [1, image.height, image.width, 3]
            input = new Tensor(image, 3);
            float[] tensor_array = input.data.Download(input.shape);
            
            // Apply preprocessing steps
            preProcessFunction(tensor_array);
            // Update input tensor with new color data
            input = new Tensor(input.shape.batch,
                               input.shape.height,
                               input.shape.width,
                               input.shape.channels,
                               tensor_array);
        }
    }

    // Obtains the model output and either decodes single or mutlple poses
    private void ProcessOutput(IWorker engine)
    {
        // Get the model output 
        // TODO: adapt resnet34 to fit here for 3D estimation
        Tensor heatmaps2D = engine.PeekOutput(predictionLayer);
        Tensor offsets2D = engine.PeekOutput(offsetsLayer);
        Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
        Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

        //Debug.Log(heatmaps2D.shape);
        //Debug.Log(offsets2D.shape);

        // Calculate the stride used to scale down the inputImage
        int stride = (imageDims.y - 1) / (heatmaps2D.shape.height - 1);
        stride -= (stride % 8);

        if (estimationType == EstimationType.SinglePose)
        {
            // Initialize the array of Keypoint arrays
            if (poses == null)
            {
                //Initialize that damn previous pose point
                previous_poses = new Utils.Keypoint[1][];
                previous_poses[0] = new Utils.Keypoint[heatmaps2D.channels];
                for (int c = 0; c < heatmaps2D.channels; c++)
                {
                    previous_poses[0][c] = new Utils.Keypoint();
                }
            }
            else
            {
                previous_poses = poses;
            }

            poses = new Utils.Keypoint[1][];

            bool reorder = false;
            
            // OpenPose use a different number of keypoints in a different order
            if (modelType == ModelType.OpenPoseSinglePoseOnly)
            {
                reorder = true;
            }

            // Determine the key point locations
            poses[0] = Utils.DecodeSinglePose(heatmaps2D, offsets2D, stride, previous_poses[0], UseKalmanFiltering, KalmanParamQ, KalmanParamR, reorder);
        }
        else
        {
            // Because I idotly creates the keypoints in that fonction, I need to create the "previous_poses" outside.
            // Bad design Mike, bad design.

            // Initialize the array of Keypoint arrays
            if (poses == null)
            {
                //Initialize that damn previous pose point
                poses = new Utils.Keypoint[maxPoses][];
                poses[0] = new Utils.Keypoint[heatmaps2D.channels];
                previous_poses = new Utils.Keypoint[maxPoses][];
                previous_poses[0] = new Utils.Keypoint[heatmaps2D.channels];
                for (int a = 0; a < maxPoses; a++)
                {
                    poses[a] = new Utils.Keypoint[heatmaps2D.channels]; //I think it's too much.
                    previous_poses[a] = new Utils.Keypoint[heatmaps2D.channels]; //I think it's too much.
                    for (int c = 0; c < heatmaps2D.channels; c++)
                    {
                        poses[0][c] = new Utils.Keypoint();
                        previous_poses[0][c] = new Utils.Keypoint();
                    }
                }
            }
            else
            {
                for (int a = 0; a < poses.Length; a++)
                {
                    for (int c = 0; c < poses[a].Length; c++)
                    {
                        previous_poses[a][c] = poses[a][c];
                    }
                }
            }

            //Initialize that pose point
            poses = new Utils.Keypoint[maxPoses][];
            poses[0] = new Utils.Keypoint[heatmaps2D.channels];
            for (int a = 0; a < maxPoses; a++)
            {
                poses[a] = new Utils.Keypoint[heatmaps2D.channels]; //I think it's too much.
                for (int c = 0; c < heatmaps2D.channels; c++)
                {
                    poses[0][c] = new Utils.Keypoint();
                 }
            }

            // Determine the key point locations
            Utils.DecodeMultiplePoses(
                heatmaps2D, offsets2D,
                displacementFWD, displacementBWD,
                stride: stride, maxPoseDetections: maxPoses,
                cur_poses: poses,
                pred: previous_poses, Use_Kalman: UseKalmanFiltering, kalman_Q: KalmanParamQ, kalman_R: KalmanParamR,
                scoreThreshold: scoreThreshold,
                nmsRadius: nmsRadius);

        }

        heatmaps2D.Dispose();
        offsets2D.Dispose();
        displacementFWD.Dispose();
        displacementBWD.Dispose();
    }

    // Update is called once per frame
    void Update()
    {

        // Prevent the input dimensions from going too low for the model
        imageDims.x = Mathf.Max(imageDims.x, 64);
        imageDims.y = Mathf.Max(imageDims.y, 64);

        // Update the input dimensions while maintaining the source aspect ratio
        if (imageDims.x != targetDims.x)
        {
            aspectRatioScale = (float)videoTexture.height / videoTexture.width;
            targetDims.y = (int)(imageDims.x * aspectRatioScale);
            imageDims.y = targetDims.y;
            targetDims.x = imageDims.x;
        }
        if (imageDims.y != targetDims.y)
        {
            aspectRatioScale = (float)videoTexture.width / videoTexture.height;
            targetDims.x = (int)(imageDims.y * aspectRatioScale);
            imageDims.x = targetDims.x;
            targetDims.y = imageDims.y;
        }

        // Update the rTex dimensions to the new input dimensions
        if (imageDims.x != rTex.width || imageDims.y != rTex.height)
        {
            RenderTexture.ReleaseTemporary(rTex);
            // Assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, rTex.format);
        }

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(videoTexture, rTex);
        ProcessImage(rTex);

        // Reinitialize Barracuda with the selected model and backend 
        if (engine.modelType != modelType || engine.workerType != workerType)
        {
            engine.worker.Dispose();
            StartEngine();
        }

        // Execute neural network with the provided input
        engine.worker.Execute(input);
        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Decode the keypoint coordinates from the model output
        ProcessOutput(engine.worker);

        // Reinitialize pose skeletons
        if (maxPoses != skeletons.Length)
        {
            foreach (PoseSkeleton skeleton in skeletons)
            {
                skeleton.Cleanup();
            }

            // Initialize pose skeletons
            DrawSkeletons(); 
        }

        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

        // Update the pose skeletons
        for (int i = 0; i < skeletons.Length; i++)
        {
            if (i <= poses.Length - 1)
            {
                skeletons[i].ToggleSkeleton(true);

                // Update the positions for the key point GameObjects
                skeletons[i].UpdateKeyPointPositions(poses[i], scale, videoTexture, minConfidence);
                skeletons[i].UpdateLines();
            }
            else
            {
                skeletons[i].ToggleSkeleton(false);
            }
        }
    }

    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        engine.worker.Dispose();
    }
}
