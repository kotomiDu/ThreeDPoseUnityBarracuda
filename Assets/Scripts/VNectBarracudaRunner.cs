using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;

using System;
using System.Runtime.InteropServices;
using System.IO;


/// <summary>
/// Define Joint points
/// </summary>
public class VNectBarracudaRunner : MonoBehaviour
{
    /// <summary>
    /// Neural network model
    /// </summary>
    public NNModel NNModel;

    public WorkerFactory.Type WorkerType = WorkerFactory.Type.Auto;
    public bool Verbose = true;

    public VNectModel VNectModel;

    public VideoCapture videoCapture;

    private Model _model;
    private IWorker _worker;

    /// <summary>
    /// Coordinates of joint points
    /// </summary>
    private VNectModel.JointPoint[] jointPoints;
    
    /// <summary>
    /// Number of joint points
    /// </summary>
    private const int JointNum = 24;

    /// <summary>
    /// input image size
    /// </summary>
    public int InputImageSize;

    /// <summary>
    /// input image size (half)
    /// </summary>
    private float InputImageSizeHalf;

    /// <summary>
    /// column number of heatmap
    /// </summary>
    public int HeatMapCol;
    private float InputImageSizeF;

    /// <summary>
    /// Column number of heatmap in 2D image
    /// </summary>
    private int HeatMapCol_Squared;
    
    /// <summary>
    /// Column nuber of heatmap in 3D model
    /// </summary>
    private int HeatMapCol_Cube;
    private float ImageScale;

    /// <summary>
    /// Buffer memory has 2D heat map
    /// </summary>
    private float[] heatMap2D;

    /// <summary>
    /// Buffer memory has offset 2D
    /// </summary>
    private float[] offset2D;
    
    /// <summary>
    /// Buffer memory has 3D heat map
    /// </summary>
    private float[] heatMap3D;
    
    /// <summary>
    /// Buffer memory hash 3D offset
    /// </summary>
    private float[] offset3D;
    private float unit;
    
    /// <summary>
    /// Number of joints in 2D image
    /// </summary>
    private int JointNum_Squared = JointNum * 2;
    
    /// <summary>
    /// Number of joints in 3D model
    /// </summary>
    private int JointNum_Cube = JointNum * 3;

    /// <summary>
    /// HeatMapCol * JointNum
    /// </summary>
    private int HeatMapCol_JointNum;

    /// <summary>
    /// HeatMapCol * JointNum_Squared
    /// </summary>
    private int CubeOffsetLinear;

    /// <summary>
    /// HeatMapCol * JointNum_Cube
    /// </summary>
    private int CubeOffsetSquared;

    /// <summary>
    /// For Kalman filter parameter Q
    /// </summary>
    public float KalmanParamQ;

    /// <summary>
    /// For Kalman filter parameter R
    /// </summary>
    public float KalmanParamR;

    /// <summary>
    /// Lock to update VNectModel
    /// </summary>
    private bool Lock = true;

    /// <summary>
    /// Use low pass filter flag
    /// </summary>
    public bool UseLowPassFilter;

    /// <summary>
    /// For low pass filter
    /// </summary>
    public float LowPassParam;

    public Text Msg;
    public float WaitTimeModelLoad = 10f;
    private float Countdown = 0;
    public Texture2D InitImg;

    //Lets make our calls from the Plugin
    [DllImport("network.dll", EntryPoint = "createModel")]
    private static extern IntPtr createModel();
    [DllImport("network.dll", EntryPoint = "initModel")]
    private static extern bool initModel(IntPtr model, string modelpath, string device);
    [DllImport("network.dll", EntryPoint = "inferModel")]
    private static extern bool inferModel(IntPtr model
        , IntPtr texture1, IntPtr texture2, IntPtr texture3, int width, int height
        ,  int offset3Dsize, int heatmap3Dsize,out IntPtr offset3Dresult, out IntPtr heatmap3Dresult);

    private IntPtr context;
    int offset3DSize = 2016 * 28 * 28;
    int heatMap3DSize = 672 * 28 * 28;
    IntPtr offset3DPtr, heatMap3DPtr;
    IntPtr ovinput;

    private void Start()
    {
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
        InputImageSizeF = InputImageSize;
        InputImageSizeHalf = InputImageSizeF / 2f;
        ImageScale = InputImageSize / (float)HeatMapCol;// 224f / (float)InputImageSize;

        // Disabel sleep
        Screen.sleepTimeout = SleepTimeout.NeverSleep;

        // Init model
        //_model = ModelLoader.Load(NNModel, Verbose);
        //_worker = WorkerFactory.CreateWorker(WorkerType, _model, Verbose);

        //????  Init openvino Model
        context = createModel();
        string modelPath = Application.dataPath + "/Scripts/Model/OV_FP32/Resnet34_3inputs_448x448_20200609.xml";
        bool success = initModel(context, modelPath, "CPU");
        Debug.Log(success);

        StartCoroutine("WaitLoad");
    }

    private void Update()
    {
      
        if (!Lock)
        {
            UpdateVNectModel();
        }
    }

    private IEnumerator WaitLoad()
    {
        /* inputs[inputName_1] = new Tensor(InitImg);
         inputs[inputName_2] = new Tensor(InitImg);
         inputs[inputName_3] = new Tensor(InitImg);

         // Create input and Execute model
         yield return _worker.StartManualSchedule(inputs);

         // Get outputs
         for (var i = 2; i < _model.outputs.Count; i++)
         {
             b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
         }

         // Get data from outputs
         offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
         heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);

         // Release outputs
         for (var i = 2; i < b_outputs.Length; i++)
         {
             b_outputs[i].Dispose();
         }*/

        // openvino execution
        Texture2D readableTex = duplicateTexture(InitImg);

        bool sucess = inferModel(context
        , getTexPtr(readableTex)
        , getTexPtr(readableTex)
        , getTexPtr(readableTex)
        , InputImageSize
        , InputImageSize
        , offset3DSize
        , heatMap3DSize
        , out offset3DPtr
        , out heatMap3DPtr);
       
        Marshal.Copy(offset3DPtr, offset3D, 0, offset3DSize);
        Marshal.FreeCoTaskMem(offset3DPtr);
        Marshal.Copy(heatMap3DPtr, heatMap3D, 0, heatMap3DSize);
        Marshal.FreeCoTaskMem(heatMap3DPtr);
        //System.IO.File.WriteAllText("offset3D_openvino.txt", string.Join(" ", offset3D));
        //System.IO.File.WriteAllText("heatMap3D_openvino.txt", string.Join(" ", heatMap3D));
        // Init VNect model
        jointPoints = VNectModel.Init();

        PredictPose();

        yield return new WaitForSeconds(WaitTimeModelLoad);

        // Init VideoCapture
        videoCapture.Init(InputImageSize, InputImageSize);
        Lock = false;
        Msg.gameObject.SetActive(false);
    }

    private const string inputName_1 = "input.1";
    private const string inputName_2 = "input.4";
    private const string inputName_3 = "input.7";
    /*
    private const string inputName_1 = "0";
    private const string inputName_2 = "1";
    private const string inputName_3 = "2";
    */

    private void UpdateVNectModel()
    {
        /*
        input = new Tensor(videoCapture.MainTexture);
        if (inputs[inputName_1] == null)
        {
            inputs[inputName_1] = input;
            inputs[inputName_2] = new Tensor(videoCapture.MainTexture);
            inputs[inputName_3] = new Tensor(videoCapture.MainTexture);
        }
        else
        {
            inputs[inputName_3].Dispose();

            inputs[inputName_3] = inputs[inputName_2];
            inputs[inputName_2] = inputs[inputName_1];
            inputs[inputName_1] = input;
        }*/
        
        ovinput = getTexPtr(toTexture2D(videoCapture.MainTexture));

        //Debug.Log("Test" + ovinputs[inputName_1]);
        if (ovinputs[inputName_1] .Equals( IntPtr.Zero))
        {
            Debug.Log("test1");
            ovinputs[inputName_1] = ovinput;
            ovinputs[inputName_2] = getTexPtr(toTexture2D(videoCapture.MainTexture));
            ovinputs[inputName_3] = getTexPtr(toTexture2D(videoCapture.MainTexture));
        }
        else
        {
            //ovinputs[inputName_3].Dispose();

            ovinputs[inputName_3] = ovinputs[inputName_2];
            ovinputs[inputName_2] = ovinputs[inputName_1];
            ovinputs[inputName_1] = ovinput;
        }

      
        StartCoroutine(ExecuteModelAsync());
    }

    /// <summary>
    /// Tensor has input image
    /// </summary>
    /// <returns></returns>
    Tensor input = new Tensor();
    Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>() { { inputName_1, null }, { inputName_2, null }, { inputName_3, null }, };
    Tensor[] b_outputs = new Tensor[4];

    /// <summary>
    /// OpenVINO input image
    /// </summary>
    /// <returns></returns>
    
    Dictionary<string, IntPtr> ovinputs = new Dictionary<string, IntPtr>() { { inputName_1, IntPtr.Zero }, { inputName_2, IntPtr.Zero }, { inputName_3, IntPtr.Zero }, };

    private IEnumerator ExecuteModelAsync()
    {
        /*
         * // Create input and Execute model
        yield return _worker.StartManualSchedule(inputs);

        // Get outputs
        for (var i = 2; i < _model.outputs.Count; i++)
        {
            b_outputs[i] = _worker.PeekOutput(_model.outputs[i]);
        }

       
        // Get data from outputs
        offset3D = b_outputs[2].data.Download(b_outputs[2].shape);
        heatMap3D = b_outputs[3].data.Download(b_outputs[3].shape);

        // Release outputs
        for (var i = 2; i < b_outputs.Length; i++)
        {
            b_outputs[i].Dispose();
        }
        */
       
      inferModel(context
      , ovinputs[inputName_1]
      , ovinputs[inputName_2]
      , ovinputs[inputName_3]
      , InputImageSize
      , InputImageSize
      , offset3DSize
      , heatMap3DSize
      , out offset3DPtr
      , out heatMap3DPtr);

        if(offset3DPtr != IntPtr.Zero && heatMap3DPtr != IntPtr.Zero){
            Marshal.Copy(offset3DPtr
            , offset3D
            , 0
            , offset3DSize);
            Marshal.FreeCoTaskMem(offset3DPtr);
            Marshal.Copy(heatMap3DPtr
                , heatMap3D
                , 0
                , heatMap3DSize);
            Marshal.FreeCoTaskMem(heatMap3DPtr);
        }

        Debug.Log(heatMap3D[1000]);
        yield return new WaitForSeconds(WaitTimeModelLoad);

        PredictPose();
    }

    /// <summary>
    /// Predict positions of each of joints based on network
    /// </summary>
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

        // Kalman filter
        foreach (var jp in jointPoints)
        {
            KalmanUpdate(jp);
        }

        // Low pass filter
        if (UseLowPassFilter)
        {
            foreach (var jp in jointPoints)
            {
                jp.PrevPos3D[0] = jp.Pos3D;
                for (var i = 1; i < jp.PrevPos3D.Length; i++)
                {
                    jp.PrevPos3D[i] = jp.PrevPos3D[i] * LowPassParam + jp.PrevPos3D[i - 1] * (1f - LowPassParam);
                }
                jp.Pos3D = jp.PrevPos3D[jp.PrevPos3D.Length - 1];
            }
        }
    }

    /// <summary>
    /// Kalman filter
    /// </summary>
    /// <param name="measurement">joint points</param>
    void KalmanUpdate(VNectModel.JointPoint measurement)
    {
        measurementUpdate(measurement);
        measurement.Pos3D.x = measurement.X.x + (measurement.Now3D.x - measurement.X.x) * measurement.K.x;
        measurement.Pos3D.y = measurement.X.y + (measurement.Now3D.y - measurement.X.y) * measurement.K.y;
        measurement.Pos3D.z = measurement.X.z + (measurement.Now3D.z - measurement.X.z) * measurement.K.z;
        measurement.X = measurement.Pos3D;
    }

	void measurementUpdate(VNectModel.JointPoint measurement)
    {
        measurement.K.x = (measurement.P.x + KalmanParamQ) / (measurement.P.x + KalmanParamQ + KalmanParamR);
        measurement.K.y = (measurement.P.y + KalmanParamQ) / (measurement.P.y + KalmanParamQ + KalmanParamR);
        measurement.K.z = (measurement.P.z + KalmanParamQ) / (measurement.P.z + KalmanParamQ + KalmanParamR);
        measurement.P.x = KalmanParamR * (measurement.P.x + KalmanParamQ) / (KalmanParamR + measurement.P.x + KalmanParamQ);
        measurement.P.y = KalmanParamR * (measurement.P.y + KalmanParamQ) / (KalmanParamR + measurement.P.y + KalmanParamQ);
        measurement.P.z = KalmanParamR * (measurement.P.z + KalmanParamQ) / (KalmanParamR + measurement.P.z + KalmanParamQ);
    }


    Texture2D toTexture2D(RenderTexture texture)
    {
       // Texture2D tex = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
        Texture2D tex = new Texture2D(texture.width, texture.height, TextureFormat.ARGB32, false);
        // ReadPixels looks at the active RenderTexture.
        RenderTexture.active = texture;
        tex.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
        tex.Apply();
        //File.WriteAllBytes(Application.persistentDataPath + "/" + fileindex + "pose.png", (byte[])tex.EncodeToPNG());
        //fileindex++;
        return tex;
    }
    private int fileindex = 0;

    IntPtr getTexPtr(Texture2D tex)
    {
        Color32[] pixel32 = tex.GetPixels32();
        //Pin pixel32 array
        GCHandle pixelHandle = GCHandle.Alloc(pixel32, GCHandleType.Pinned);
        //Get the pinned address
        return pixelHandle.AddrOfPinnedObject();
    }

    Texture2D duplicateTexture(Texture2D source)
    {
        byte[] pix = source.GetRawTextureData();
        Texture2D readableText = new Texture2D(source.width, source.height, source.format, false);
        readableText.LoadRawTextureData(pix);
        readableText.Apply();
        return readableText;
    }
}
