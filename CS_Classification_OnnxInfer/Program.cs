using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using static System.Collections.Specialized.BitVector32;
using System.ComponentModel;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using System.Security.Cryptography;
using System.Threading;

namespace Classification_OnnxInfer
{
    class Program
    {
        static List<int> predictions = new List<int>();
        static List<float> confidences = new List<float>();

        public static void Main(string[] args)
        {
            var onnx = createSession(
                @"D:\software\PyCharm Community Edition 2023.1.2\deep learning\MNIST_Demo\checkpoints\mnist.onnx"
            );
            PrintInputMetadata(onnx.InputMetadata);

            Stopwatch sw = Stopwatch.StartNew();

            var test_images = Directory.GetFiles(
                @"D:\software\PyCharm Community Edition 2023.1.2\deep learning\MNIST_Demo\test_images\2"
            );

            var list_images = test_images.Select(s =>
            {
                var image = new Mat(s, ImreadModes.Grayscale);

                var dst = new Mat();
                Cv2.Resize(
                    image,
                    dst,
                    new OpenCvSharp.Size(32, 32),
                    0,
                    0,
                    InterpolationFlags.Cubic
                );
                var bitmap = BitmapConverter.ToBitmap(dst);

                return bitmap;
            });
            confidences.Clear();
            sw.Restart();
            foreach (var item in list_images)
            {
                var arr = ConvertImageThreeChannelToFloatTensorUnsafe(item);
                inferOnnx(onnx, arr);
            }
            sw.Stop();
            list_images.ToList().ForEach(s => s.Dispose());
            Console.WriteLine(
                $"average confidence from local:{confidences.Average()},count:{confidences.Count},[{predictions.Where(s => s == 2).Count()}/{list_images.ToList().Count}]"
            );
            sw.Stop();
            onnx.Dispose();
            Console.ReadKey();
        }

        static InferenceSession createSession(string modelpath)
        {
            string modelPath = modelpath;

            // Optional : Create session options and set the graph optimization level for the session
            //SessionOptions options = new SessionOptions();
            //options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            //using (var session = new InferenceSession(modelPath, options))
            var session = new InferenceSession(modelPath);

            return session;
        }

        static void inferOnnx(InferenceSession onnx, List<float[]> imageDatas)
        {
            Stopwatch sw = Stopwatch.StartNew();
            IReadOnlyDictionary<string, NodeMetadata> inputMeta = onnx.InputMetadata;
            for (int i = 0; i < imageDatas.Count; i++)
            {
                List<NamedOnnxValue> container = new List<NamedOnnxValue>();
                foreach (var name in inputMeta.Keys)
                {
                    //float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model
                    //图像输入
                    float[] inputData = imageDatas[i];
                    //图像标签
                    //string label = Utilities.ImageLabels[imageIndex];
                    //Console.WriteLine("Selected image is the number: " + label);
                    //onnx的 inputMeta，包含了输入的信息，比如维度【1，784】，类型等

                    //onnx的 container，用来存放输入数据

                    //PrintInputMetadata(inputMeta);
                    //图像转tensor
                    sw.Restart();
                    DenseTensor<float> tensor = new DenseTensor<float>(
                        inputData,
                        inputMeta[name].Dimensions
                    );
                    sw.Stop();
                    //Logger.Log("转换tensor耗时" + sw.ElapsedMilliseconds + "ms");
                    //图像丢给onnx的container，根据inputMeta[name]的信息，onnx会把tensor转成onnx的格式
                    //[1,784]
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                sw.Restart();
                using (var results = onnx.Run(container)) // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    sw.Stop();
                    //Logger.Log("推理耗时" + sw.ElapsedMilliseconds + "ms");
                    // Get the results
                    foreach (DisposableNamedOnnxValue r in results)
                    {
                        //Console.WriteLine("Output Name: {0}", r.Name);
                        int prediction = MaxProbability(r.AsTensor<float>(), out var confidence);
                        //Console.WriteLine($"confidence：{confidence}");
                        //Console.WriteLine("Prediction: " + prediction.ToString());
                        predictions.Add(prediction);
                        confidences.Add(confidence);
                        //Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
                container.Clear();
            }
        }

        static void inferOnnx(InferenceSession onnx, Tensor<float> imageData)
        {
            Stopwatch sw = Stopwatch.StartNew();
            IReadOnlyDictionary<string, NodeMetadata> inputMeta = onnx.InputMetadata;

            List<NamedOnnxValue> container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                container.Add(NamedOnnxValue.CreateFromTensor<float>(name, imageData));

                // Run the inference
                sw.Restart();
                //[1,28,28,1],不行
                using (var results = onnx.Run(container)) // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    sw.Stop();
                    //Logger.Log("推理耗时" + sw.ElapsedMilliseconds + "ms");
                    // Get the results
                    foreach (DisposableNamedOnnxValue r in results)
                    {
                        //Console.WriteLine("Output Name: {0}", r.Name);
                        int prediction = MaxProbability(r.AsTensor<float>(), out var confidence);
                        double[] buff = new double[5];
                        int maxIndex = int.MinValue;
                        double maxPro = double.MinValue;
                        SoftMax(
                            r.AsEnumerable<float>().ToArray(),
                            ref buff,
                            ref maxIndex,
                            ref maxPro
                        );
                        //Console.WriteLine($"confidence：{confidence}");
                        Console.WriteLine(
                            "Prediction: " + prediction.ToString() + $"confidence：{(float)maxPro}"
                        );
                        predictions.Add(prediction);
                        confidences.Add((float)maxPro);
                        //Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
                container.Clear();
            }
        }

        /// <summary>
        /// 获取tensor的最大值，相当于获取预测结果，torch.argmax()
        /// </summary>
        /// <param name="probabilities"></param>
        /// <returns></returns>
        static int MaxProbability(Tensor<float> probabilities, out float confidence)
        {
            confidence = 0;
            float max = -9999.9F;
            int maxIndex = -1;
            for (int i = 0; i < probabilities.Length; ++i)
            {
                float prob = probabilities.GetValue(i);
                if (prob > max)
                {
                    max = prob;
                    maxIndex = i;
                    confidence = max;
                }
            }
            return maxIndex;
        }

        private static void SoftMax(
            float[] pro,
            ref double[] proExp,
            ref int maxIndex,
            ref double maxPro
        )
        {
            if (pro != null && proExp != null && pro.Length == proExp.Length)
            {
                double sumExp = 0;
                for (int i = 0; i < pro.Length; i++)
                {
                    proExp[i] = System.Math.Exp(pro[i]);
                    sumExp += proExp[i];
                }

                maxPro = double.MinValue;
                double tempNowSoftMax = 0;
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

        static void PrintInputMetadata(IReadOnlyDictionary<string, NodeMetadata> inputMeta)
        {
            foreach (var name in inputMeta.Keys)
            {
                for (int i = 0; i < inputMeta[name].Dimensions.Length; ++i)
                {
                    Console.WriteLine(inputMeta[name].Dimensions[i]);
                }
            }
        }

        public static Tensor<float> ConvertImageThreeChannelToFloatTensorUnsafe(Bitmap image)
        {
            // Create the Tensor with the appropiate dimensions  for the NN
            Tensor<float> data = new DenseTensor<float>(new[] { 1, 3, image.Width, image.Height });

            BitmapData bmd = image.LockBits(
                new Rectangle(0, 0, image.Width, image.Height),
                System.Drawing.Imaging.ImageLockMode.ReadOnly,
                image.PixelFormat
            );

            int PixelSize = 3;
            unsafe
            {
                for (int y = 0; y < bmd.Height; y++)
                {
                    // row is a pointer to a full row of data with each of its colors
                    byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                    for (int x = 0; x < bmd.Width; x++)
                    {
                        data[0, 0, y, x] = row[x * PixelSize + 2] / (float)255.0;
                        data[0, 1, y, x] = row[x * PixelSize + 1] / (float)255.0;
                        data[0, 2, y, x] = row[x * PixelSize + 0] / (float)255.0;
                    }
                }

                image.UnlockBits(bmd);
            }
            return data;
        }

        public static Tensor<float> ConvertImageOneChannelToFloatTensorUnsafe(Bitmap image)
        {
            Tensor<float> data = new DenseTensor<float>(new[] { 1, image.Width * image.Height });

            BitmapData bmd = image.LockBits(
                new Rectangle(0, 0, image.Width, image.Height),
                System.Drawing.Imaging.ImageLockMode.ReadOnly,
                image.PixelFormat
            );
            unsafe
            {
                int n = 0;
                for (int y = 0; y < bmd.Height; y++)
                {
                    byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                    for (int x = 0; x < bmd.Width; x++)
                    {
                        data[0, n++] = row[x];
                    }
                }

                image.UnlockBits(bmd);
            }
            return data;
        }
    }
}
