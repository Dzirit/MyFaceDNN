using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace MyFaceDNN
{
    public class FaceProcessHaar
    {
        private CascadeClassifier haarCascade;
        private Image<Gray, Byte> detectedFace = null;
        private List<FaceData> faceList = new List<FaceData>();
        private VectorOfMat imageList = new VectorOfMat();
        private List<string> nameList = new List<string>();
        private VectorOfInt labelList = new VectorOfInt();
        private EigenFaceRecognizer recognizer;
        public static string FacePhotosPath = "Database\\Faces\\";
        public static string FaceListTextFile = "Database\\FaceList.txt";
        public static string HaarCascadePath = "haarcascade_frontalface_default.xml";
        public static string ImageFileExtension = ".bmp";
        private string faceName;
        public string FaceName
        {
            get { return faceName; }
            set
            {
                faceName = value.ToUpper();
            }
        }
        public FaceProcessHaar()
        {
            GetFacesList();
        }
        private void GetFacesList()
        {
            //haar cascade classifier
            if (!File.Exists(HaarCascadePath))
            {
                Console.WriteLine( $"Cannot find Haar cascade data file:{HaarCascadePath}");
            }

            haarCascade = new CascadeClassifier(HaarCascadePath);
            faceList.Clear();
            string line;
            FaceData faceInstance = null;

            // Create empty directory / file for face data if it doesn't exist
            if (!Directory.Exists(FacePhotosPath))
            {
                Directory.CreateDirectory(FacePhotosPath);
            }

            if (!File.Exists(FaceListTextFile))
            {
                string text = "Cannot find face data file:\n\n";
                text += FaceListTextFile + "\n\n";
                text += "If this is your first time running the app, an empty file will be created for you.";
                Console.WriteLine(text);
                String dirName = Path.GetDirectoryName(FaceListTextFile);
                Directory.CreateDirectory(dirName);
                File.Create(FaceListTextFile).Close();
            }

            StreamReader reader = new StreamReader(FaceListTextFile);
            int i = 0;
            while ((line = reader.ReadLine()) != null)
            {
                string[] lineParts = line.Split(':');
                faceInstance = new FaceData();
                faceInstance.FaceImage = new Image<Gray, byte>(FacePhotosPath + lineParts[0] + ImageFileExtension);
                faceInstance.PersonName = lineParts[1];
                faceList.Add(faceInstance);
            }
            foreach (var face in faceList)
            {
                imageList.Push(face.FaceImage.Mat);
                nameList.Add(face.PersonName);
                labelList.Push(new[] { i++ });
            }
            reader.Close();

            // Train recogniser
            if (imageList.Size > 0)
            {
                recognizer = new EigenFaceRecognizer(imageList.Size);
                recognizer.Train(imageList, labelList);
            }

        }

        public void ProcessFrame(ref Mat frame)
        {
            var bgrFrame = frame.ToImage<Bgr, Byte>();

            if (bgrFrame != null)
            {
                try
                {//for emgu cv bug
                    Image<Gray, byte> grayframe = bgrFrame.Convert<Gray, byte>();

                    Rectangle[] faces = haarCascade.DetectMultiScale(grayframe, 1.2, 10, new System.Drawing.Size(50, 50), new System.Drawing.Size(200, 200));

                    //detect face
                    FaceName = "No face detected";
                    foreach (var face in faces)
                    {
                        //CvInvoke.Rectangle(frame, face, new MCvScalar(255, 0, 0));
                        bgrFrame.Draw(face, new Bgr(255, 255, 0), 1);
                        detectedFace = bgrFrame.Copy(face).Convert<Gray, byte>();
                        var name=FaceRecognition();
                        if (name != null)
                            CvInvoke.PutText(bgrFrame, name, new Point(10, bgrFrame.Height-20), FontFace.HersheyComplex, 4.0, new Bgr(Color.Black).MCvScalar);
                        bgrFrame.Mat.CopyTo(frame);
                        break;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                }
                bgrFrame.Dispose();
            }
        }


        private string FaceRecognition()
        {
            if (imageList.Size != 0)
            {
                //Eigen Face Algorithm
                FaceRecognizer.PredictionResult result = recognizer.Predict(detectedFace.Resize(100, 100, Inter.Cubic));
                FaceName = nameList[result.Label];
                if (result.Distance!=0)
                    Console.WriteLine($"Distance:{result.Distance}");
                Console.SetCursorPosition(0, 1);
            }
            else
            {
                FaceName = null;
            }
            return FaceName;
        }
    }
}
