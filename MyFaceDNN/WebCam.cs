using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Text;

namespace MyFaceDNN
{
    public class WebCam
    {
        static string windowName = "1";
        public WebCam(string camIndex = "0")
        {
            var faceDetection = new FaceProccesDNN();
            var faceRecognition = new FaceProcessHaar();
            VideoCapture videoCapture = new VideoCapture(int.Parse(camIndex));
   
            
            CvInvoke.NamedWindow(windowName, Emgu.CV.CvEnum.WindowFlags.AutoSize);
            while (/*CvInvoke.WaitKey(1) != 27 */CvInvoke.GetWindowProperty(windowName, Emgu.CV.CvEnum.WindowPropertyFlags.AutoSize) >= 0)//Esc
            {
                try
                {
                    var frame = videoCapture.QueryFrame();
                    //faceDetection.DetectAndRender(frame);
                    faceRecognition.ProcessFrame(ref frame);
                    CvInvoke.Imshow(windowName, frame);
                    frame.Dispose();
                    CvInvoke.WaitKey(1);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                }
                //faceDetection.DetectAndRender(frame);
                //CvInvoke.Imshow(windowName, frame);
                //frame.Dispose();
                //CvInvoke.WaitKey(5);
            }
        }
    }
}
