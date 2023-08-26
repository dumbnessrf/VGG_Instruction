using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;

namespace OpencvDnn_Infer_Onnx
{
    public partial class FrmMain : Form
    {
        public Param m_Param = new Param();
        public List<EvalResult> m_InferResults = new List<EvalResult>();
        private readonly string onnxFile =
            @"D:\\software\\PyCharm Community Edition 2023.1.2\\deep learning\\MNIST_Demo\\checkpoints\\mnist.onnx";
        OpenCvSharp.Dnn.Net net = null;

        public FrmMain()
        {
            InitializeComponent();
            m_Param.InitImagesFromRoot(
                @"D:\software\PyCharm Community Edition 2023.1.2\deep learning\MNIST_Demo\test_images"
            );
            propertyGrid1.SelectedObject = m_Param;
        }

        private void buttonTest_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                m_InferResults.Clear();
                for (int i = 0; i < m_Param.Images.Count; i++)
                {
                    var res = Eval(m_Param.Images[i]);

                    m_InferResults.Add(res);
                }
                var pros = m_InferResults.Where(d => d.Result).Select(s => s.Probability).ToList();
                m_Param.CorrectMaxConfidence = pros.Max();
                m_Param.CorrectMinConfidence = pros.Min();
                m_Param.CorrectAverageConfidence = pros.Average();

                m_Param.CorrectNum = pros.Count();
                m_Param.TotalNum = m_Param.Images.Count;
                m_Param.CorrectProbability =
                    Convert.ToSingle(m_Param.CorrectNum) / m_Param.TotalNum;
                this.BeginInvoke(
                    new MethodInvoker(() =>
                    {
                        propertyGrid1.Refresh();
                    })
                );
            }).Start();
        }

        private EvalResult Eval(ImageData img_data)
        {
            EvalResult inferResult = new EvalResult();
            inferResult.Image_Path = img_data.Image_Path;
            try
            {
                if (net != null && !net.Empty())
                {
                    UpdateImage(img_data.Image);
                    var inputBlob = new Mat();
                    Cv2.Resize(
                        img_data.Image,
                        inputBlob,
                        new OpenCvSharp.Size(32, 32),
                        0,
                        0,
                        InterpolationFlags.Cubic
                    );

                    inputBlob = CvDnn.BlobFromImage(
                        inputBlob,
                        scaleFactor: 1.0 / 255.0, //不加的话，softmax的exp会得到+∞或-∞，需要normalize到0~1
                        swapRB: true //opencv的imread默认是bgr，调换rb
                    );

                    // Set input blob
                    net.SetInput(inputBlob);

                    // Make forward pass
                    var output = net.Forward("output");

                    float[] buff = new float[output.Rows * output.Cols];
                    System.Runtime.InteropServices.Marshal.Copy(
                        output.Ptr(0),
                        buff,
                        0,
                        buff.Length
                    );

                    float[] proBuff = new float[buff.Length];
                    int maxIndex = int.MinValue;
                    float maxPro = float.MinValue;
                    SoftMax(buff, ref proBuff, ref maxIndex, ref maxPro);
                    inferResult.Probability = maxPro;
                    inferResult.Index = maxIndex;
                    inferResult.EvalLabel = m_Param.Labels[maxIndex];
                    inferResult.RealLabel = img_data.Label;
                    if (inferResult.EvalLabel != inferResult.RealLabel)
                    {
                        inferResult.Result = false;
                    }
                    else
                    {
                        inferResult.Result = true;
                    }

                    output.Dispose();
                    inputBlob.Dispose();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"{ex.Message}{Environment.NewLine + Environment.NewLine}{ex.ToString()}"
                );
            }
            return inferResult;
        }

        private static void SoftMax(
            float[] pro,
            ref float[] proExp,
            ref int maxIndex,
            ref float maxPro
        )
        {
            if (pro != null && proExp != null && pro.Length == proExp.Length)
            {
                float sumExp = 0;
                for (int i = 0; i < pro.Length; i++)
                {
                    proExp[i] = (float)System.Math.Exp(pro[i]);
                    sumExp += proExp[i];
                }

                maxPro = float.MinValue;
                float tempNowSoftMax = 0;
                for (int i = 0; i < proExp.Length; i++)
                {
                    tempNowSoftMax = proExp[i] / sumExp;

                    proExp[i] = tempNowSoftMax;
                    if (maxPro < tempNowSoftMax)
                    {
                        maxPro = tempNowSoftMax;
                        maxIndex = i;
                    }
                }
            }
        }

        private void buttonInitModel_Click(object sender, EventArgs e)
        {
            if (net == null || net.Empty())
            {
                net = OpenCvSharp.Dnn.Net.ReadNetFromONNX(onnxFile);
                net.SetPreferableBackend(0);
                net.SetPreferableTarget(0);
            }
            else
            {
                net.Dispose();
                net = OpenCvSharp.Dnn.Net.ReadNetFromONNX(onnxFile);
                net.SetPreferableBackend(0);
                net.SetPreferableTarget(0);
            }
        }

        private void buttonLoadData_Click(object sender, EventArgs e) { }

        private void UpdateImage(Mat image)
        {
            pictureBox1.BeginInvoke(
                new MethodInvoker(() =>
                {
                    pictureBox1.Image?.Dispose();
                    pictureBox1.Image = BitmapConverter.ToBitmap(image);
                })
            );
        }
    }

    public class Param
    {
        public float CorrectMaxConfidence { get; set; }
        public float CorrectMinConfidence { get; set; }
        public float CorrectAverageConfidence { get; set; }
        public int CorrectNum { get; set; }
        public int TotalNum { get; set; }
        public float CorrectProbability { get; set; }

        [Browsable(false)]
        public string TestRoot { get; set; }

        [Browsable(false)]
        public List<ImageData> Images { get; set; } = new List<ImageData>();

        [Browsable(false)]
        public Dictionary<int, string> Labels { get; set; } =
            new Dictionary<int, string>()
            {
                { 0, "0" },
                { 1, "1" },
                { 2, "2" },
                { 3, "3" },
                { 4, "4" }
            };

        public void InitImagesFromRoot(string root)
        {
            var labels = Directory.GetDirectories(root).Select(s => Path.GetFileName(s));
            foreach (var item in labels)
            {
                var folder = Path.Combine(root, item);
                foreach (var image_path in Directory.GetFiles(folder))
                {
                    var image = new OpenCvSharp.Mat(image_path, ImreadModes.Color);
                    var image_data = new ImageData(image, item, image_path);
                    Images.Add(image_data);
                }
            }
        }
    }

    public class ImageData
    {
        public string Label;
        public string Image_Path;
        public Mat Image;

        public ImageData(Mat image, string label = "", string image_Path = "")
        {
            Label = label;
            Image_Path = image_Path;
            Image = image;
        }
    }

    public class EvalResult
    {
        public int Index;
        public string EvalLabel;
        public string RealLabel;
        public string Image_Path;
        public float Probability;
        public bool Result;
    }
}
