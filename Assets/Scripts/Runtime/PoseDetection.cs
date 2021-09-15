using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Video;

namespace Pose.Detection
{
    /// <summary>
    /// Pose Detection using Unity Barracuda.
    /// 1. Use an existing pose detection model (HRNet, DeepPose etc) and convert it to onnx model.
    /// 2. Render video to a RenderTarget and use it as the input for the model.
    /// 3. Preprocess the render target before passing it to the model. You can use Compute Shaders for this.
    /// 4. Each model requires a specific color space and resolution, you need to convert the render target first.
    /// 5. The `Model` object along with `IWorker` engine classes from Barracuda package will help you with inference.
    /// 6. Convert the output key points with confidence to actual pose and joint values.
    /// 7. Drive a skeleton/gameobjects with joint values.
    /// </summary>
    public class PoseDetection : MonoBehaviour
    {
        [SerializeField] private VideoPlayer videoPlayer;
        [SerializeField] private ComputeShader preprocessingShader;
        
        [Space]
        [SerializeField] private int imageHeight = 488;
        [SerializeField] private int imageWidth = 488;
        
        [Space]
        [SerializeField] private GameObject videoQuad;

        [Space]
        [SerializeField] public float KalmanParamQ;
        [SerializeField] public float KalmanParamR ;

        public int HeatMapCol;

        public int InputImageSizeF;
            
        public NNModel NNModel;
        public VNectModel VNectModel;

        public WorkerFactory.Type WorkerType = WorkerFactory.Type.Auto;
        
        #region private
        private int _videoHeight;
        private int _videoWidth;
        private RenderTexture videoTexture;
        private RenderTexture inputTexture;

        private float InputImageSizeHalf;
        
        private Model _model;
        private IWorker _worker;
        

        private VNectModel.JointPoint[] jointPoints;
        private const int JointNum = 24;
        // Estimated 2D keypoint locations in videoTexture and their associated confidence values
        private float[][] keypointLocations = new float[numKeypoints][];
        
        //Shader Property
        private static readonly int MainTex = Shader.PropertyToID("_MainTex");

        //Heatmaps/offsets Properties
        private float[] heatMap2D;
        private float[] heatMap3D;
        private float[] offset2D;
        private float[] offset3D;
        private float unit;
        private int HeatMapCol_Squared;
        private int HeatMapCol_Cube;
        private int HeatMapCol_JointNum;
        private float ImageScale;
        private float InputImageSizeHalf;
        private float ImageScale;

        private int CubeOffsetLinear;
        private int CubeOffsetSquared;

        private const string inputName_1 = "input_frame_1";
        private const string inputName_2 = "input_frame_2";
        private const string inputName_3 = "input_frame_3";

        private Dictionary<string, Tensor> inputs;
        private Tensor[] outputs;

        // Number of joints in 2D image
        private int numKeypoints_Squared = JointNum * 2;
    
        // Number of joints in 3D model
        private int numKeypoints_Cube = JointNum * 3;

        #endregion

        
        private void Start()
        {
            _videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            _videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;
            
            // Create a new videoTexture using the current video dimensions
            videoTexture = new RenderTexture(_videoWidth, _videoHeight, 24, RenderTextureFormat.ARGB32);
            videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;
            
            //Apply Texture to Quad
            videoQuad.gameObject.GetComponent<MeshRenderer>().material.SetTexture(MainTex, videoTexture);
            videoQuad.transform.localScale = new Vector3(_videoWidth, _videoHeight, videoQuad.transform.localScale.z);
            videoQuad.transform.position = new Vector3(0 , 0, 1);
            
            //Move Camera to keep Quad in view
            var mainCamera = Camera.main;
            if (mainCamera !=null)
            {
                mainCamera.transform.position = new Vector3(0, 0, -(_videoWidth / 2));
                mainCamera.GetComponent<Camera>().orthographicSize = _videoHeight / 2;
            }

            // Initialize 
            HeatMapCol_Squared = HeatMapCol * HeatMapCol;
            HeatMapCol_Cube = HeatMapCol * HeatMapCol * HeatMapCol;
            HeatMapCol_JointNum = HeatMapCol * JointNum;
            CubeOffsetLinear = HeatMapCol * JointNum_Cube;
            CubeOffsetSquared = HeatMapCol_Squared * JointNum_Cube;

            heatMap2D = new float[JointNum * HeatMapCol_Squared];
            offset2D = new float[JointNum * HeatMapCol_Squared * 2];
            heatMap3D = new float[JointNum * HeatMapCol_Cube];
            offset3D = new float[JointNum * HeatMapCol_Cube * 3];
            unit = 1f / (float)HeatMapCol;
            InputImageSizeF = imageHeight;
            InputImageSizeHalf = InputImageSizeF / 2f;
            ImageScale = InputImageSize / (float)HeatMapCol;// 224f / (float)InputImageSize;

            inputs = new Dictionary<string, Tensor>() { { inputName_1, null }, { inputName_2, null }, { inputName_3, null }, };
            outputs = new Tensor[4];

            
            //TODO: Create Model from onnx asset and compile it to an object
            // Init model
            _model = ModelLoader.Load(NNModel, Verbose=true);
            
            //TODO: Add Layers to model
            // ???

            //TODO: Create Worker Engine
             _worker = WorkerFactory.CreateWorker(WorkerType, _model, Verbose);

            // We need to wait 3 frames before really starting, since the model has a "memory"
            // StartCoroutine("WaitLoad");
          
        }


        // Unity method that is called every tick/frame
        private void Update()
        {
            Texture2D processedImage = PreprocessTexture();
            
            //TODO: Create Tensor 
            input = new Tensor(processedImage);

            if (inputs[inputName_1] == null)
            {
                inputs[inputName_1] = input;
                inputs[inputName_2] = new Tensor(input);
                inputs[inputName_3] = new Tensor(input);
                
                // Init VNect model for pos detection  (Note: should put in start?)
                jointPoints = VNectModel.Init();
 
            }
            else  //update previous frames 
            {
                inputs[inputName_3].Dispose();

                inputs[inputName_3] = inputs[inputName_2];
                inputs[inputName_2] = inputs[inputName_1];
                inputs[inputName_1] = input;
            }
            
            //TODO: Execute Engine
            //TODO: Process Results
            StartCoroutine(ExecuteModelAsync());
            
            //TODO: Draw Skeleton
            
            //TODO: Clean up tensors and other resources
            Destroy(processedImage);
        }

        // Launch as coroutine: function that allows pausing its execution and resuming from the same point after a condition is met
        // https://gamedevbeginner.com/coroutines-in-unity-when-and-how-to-use-them/
        private IEnumerator ExecuteModelAsync()
        {
            // Create input and Execute model
            yield return _worker.StartManualSchedule(inputs);

            // Get outputs
            for (var i = 2; i < _model.outputs.Count; i++)
            {
                outputs[i] = _worker.PeekOutput(_model.outputs[i]);
            }

            // Get data from outputs
            offset3D = outputs[2].data.Download(outputs[2].shape);
            heatMap3D = outputs[3].data.Download(outputs[3].shape);
            
            // Release outputs
            for (var i = 2; i < b_outputs.Length; i++)
            {
                outputs[i].Dispose();
            }

            // ProcessResults(engine.PeekOutput(predictionLayer), engine.PeekOutput(offsetsLayer));
            PredictPose();
        }

        // Predict positions of each of joints based on network
        private void PredictPose()
        {
            for (var j = 0; j < JointNum; j++)
            {
                var maxXIndex = 0;
                var maxYIndex = 0;
                var maxZIndex = 0;
                jointPoints[j].score3D = 0.0f;
                var jj = j * HeatMapCol;
                for (var z = 0; z < HeatMapCol; z++)
                {
                    var zz = jj + z;
                    for (var y = 0; y < HeatMapCol; y++)
                    {
                        var yy = y * HeatMapCol_Squared * JointNum + zz;
                        for (var x = 0; x < HeatMapCol; x++)
                        {
                            float v = heatMap3D[yy + x * HeatMapCol_JointNum];
                            if (v > jointPoints[j].score3D)
                            {
                                jointPoints[j].score3D = v;
                                maxXIndex = x;
                                maxYIndex = y;
                                maxZIndex = z;
                            }
                        }
                    }
                }
            
                jointPoints[j].Now3D.x = (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + j * HeatMapCol + maxZIndex] + 0.5f + (float)maxXIndex) * ImageScale - InputImageSizeHalf;
                jointPoints[j].Now3D.y = InputImageSizeHalf - (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum) * HeatMapCol + maxZIndex] + 0.5f + (float)maxYIndex) * ImageScale;
                jointPoints[j].Now3D.z = (offset3D[maxYIndex * CubeOffsetSquared + maxXIndex * CubeOffsetLinear + (j + JointNum_Squared) * HeatMapCol + maxZIndex] + 0.5f + (float)(maxZIndex - 14)) * ImageScale;
            }

            // Calculate hip location
            var lc = (jointPoints[PositionIndex.rThighBend.Int()].Now3D + jointPoints[PositionIndex.lThighBend.Int()].Now3D) / 2f;
            jointPoints[PositionIndex.hip.Int()].Now3D = (jointPoints[PositionIndex.abdomenUpper.Int()].Now3D + lc) / 2f;

            // Calculate neck location
            jointPoints[PositionIndex.neck.Int()].Now3D = (jointPoints[PositionIndex.rShldrBend.Int()].Now3D + jointPoints[PositionIndex.lShldrBend.Int()].Now3D) / 2f;

            // Calculate head location
            var cEar = (jointPoints[PositionIndex.rEar.Int()].Now3D + jointPoints[PositionIndex.lEar.Int()].Now3D) / 2f;
            var hv = cEar - jointPoints[PositionIndex.neck.Int()].Now3D;
            var nhv = Vector3.Normalize(hv);
            var nv = jointPoints[PositionIndex.Nose.Int()].Now3D - jointPoints[PositionIndex.neck.Int()].Now3D;
            jointPoints[PositionIndex.head.Int()].Now3D = jointPoints[PositionIndex.neck.Int()].Now3D + nhv * Vector3.Dot(nhv, nv);

            // Calculate spine location
            jointPoints[PositionIndex.spine.Int()].Now3D = jointPoints[PositionIndex.abdomenUpper.Int()].Now3D;
        }

        private void OnDisable()
        {
            //TODO: Release the inference engine
            _worker.Dispose()

            //Release videoTexture (?)
            videoTexture.Release();
        }
     
        #region Additional Methods
        
        private void ProcessResults(Tensor heatmaps, Tensor offsets)
        {
            // Determine the estimated key point locations using the heatmaps and offsets tensors
            
            // Calculate the stride used to scale down the inputImage
            float stride = (imageHeight - 1) / (heatmaps.shape.height - 1);
            stride -= (stride % 8);
            
            int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
            int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);
            
            //Recalculate scale for the keypoints
            var scale = (float) minDimension / (float) Mathf.Min(imageWidth, imageHeight);
            var adjustedScale = (float)maxDimension / (float)minDimension;
            
            // Iterate through heatmaps
            for (int k = 0; k < numKeypoints; k++)
            {
                //Find Location of keypoint
                var locationInfo = LocateKeyPoint(heatmaps, offsets, k);
                
                // The (x, y) coordinates contains the confidence value in the current heatmap
                var coords = locationInfo.Item1;
                var offsetVector = locationInfo.Item2;
                var confidenceValue = locationInfo.Item3;
                
                //Calulate X position and Y position
                var xPos = (coords[0]*stride + offsetVector[0])*scale;
                var yPos = (imageHeight - (coords[1]*stride + offsetVector[1]))*scale;
                if (videoTexture.width > videoTexture.height) {
                    xPos *= adjustedScale;
                }
                else
                {
                    yPos *= adjustedScale;
                }
                
                keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };
            }
        }
        
        private (float[],float[],float) LocateKeyPoint(Tensor heatmaps, Tensor offsets, int i)
        {
            //Find the heatmap index that contains the highest confidence value and the associated offset vector
            var maxConfidence = 0f;
            var coords = new float[2];
            var offsetVector = new float[2];
            
            // Iterate through heatmap columns
            for (int y = 0; y < heatmaps.shape.height; y++)
            {
                // Iterate through column rows
                for (int x = 0; x < heatmaps.shape.width; x++)
                {
                    if (heatmaps[0, y, x, i] > maxConfidence)
                    {
                        maxConfidence = heatmaps[0, y, x, i];
                        coords = new float[] { x, y };
                        offsetVector = new float[]
                        {
                            offsets[0, y, x, i + numKeypoints],
                            offsets[0, y, x, i]
                        };
                    }
                }
            }
            return (coords, offsetVector, maxConfidence);
        }

        private Texture2D PreprocessTexture()
        {
            //Apply any kind of preprocessing if required - Resize, Color values scaled etc
            
            Texture2D imageTexture = new Texture2D(videoTexture.width, 
                videoTexture.height, TextureFormat.RGBA32, false);
            
            Graphics.CopyTexture(videoTexture, imageTexture);
            Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);
            Destroy(imageTexture);

            // TODO: Apply model-specific preprocessing
            // imageTexture = PreprocessNetwork(tempTex);
            
            Destroy(tempTex);
            return imageTexture;
        }

        private Texture2D PreprocessNetwork(Texture2D inputImage)
        {
            // Use Compute Shaders (GPU) to preprocess your image
            // Each model requires a specific color space - RGB 
            // Values need to scaled to what it was trained on
            
            var numthreads = 8;
            var kernelHandle = preprocessingShader.FindKernel("Preprocess");
            var rTex = new RenderTexture(inputImage.width, 
                inputImage.height, 24, RenderTextureFormat.ARGBHalf);
            rTex.enableRandomWrite = true;
            rTex.Create();
            
            preprocessingShader.SetTexture(kernelHandle, "Result", rTex);
            preprocessingShader.SetTexture(kernelHandle, "InputImage", inputImage);
            preprocessingShader.Dispatch(kernelHandle, inputImage.height
                                                       / numthreads, 
                inputImage.width / numthreads, 1);
            
            RenderTexture.active = rTex;
            Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);
            Graphics.CopyTexture(rTex, nTex);
            RenderTexture.active = null;
            
            Destroy(rTex);
            return nTex;
        }
        
        private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
        {
            RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
            RenderTexture.active = rTex;
            
            Graphics.Blit(image, rTex);
            Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);
            
            Graphics.CopyTexture(rTex, nTex);
            RenderTexture.active = null;
            
            RenderTexture.ReleaseTemporary(rTex);
            return nTex;
        }
        
        #endregion
    }
}